"""Reproducible cache/concurrency checks against a running OpenAI-compatible API."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import platform
import threading
import time
from urllib.request import Request, ProxyHandler, build_opener


def open_url(url, body=None, headers=None, timeout=300):
    request = Request(url, data=None if body is None else json.dumps(body).encode(),
                      headers={"Content-Type": "application/json", **(headers or {})})
    return build_opener(ProxyHandler({})).open(request, timeout=timeout)


def get_json(url, timeout=5):
    with open_url(url, timeout=timeout) as response:
        return json.load(response)


def events(response):
    """Parse SSE frames, ignoring comments and allowing multi-line data."""
    data = []
    for raw in response:
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if data:
                yield "\n".join(data)
                data = []
        elif line.startswith("data:"):
            data.append(line[5:].removeprefix(" "))
    if data:
        raise ValueError("stream ended inside an SSE event")


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)] if ordered else None


def run_request(url, model, case, cache, max_tokens=32, timeout=300):
    started = time.perf_counter()
    times, content, reasoning, calls = [], [], [], []
    usage, finish, done = None, None, False
    body = {"model": model, "messages": case["messages"], "temperature": 0,
            "top_p": 1, "seed": 41000, "max_tokens": max_tokens,
            "stream": True, "stream_options": {"include_usage": True}}
    with open_url(url + "/v1/chat/completions", body,
                  {"X-DSV41-Prefix-Cache": "on" if cache else "off"}, timeout) as response:
        for event in events(response):
            if event == "[DONE]":
                done = True
                break
            item = json.loads(event)
            if "error" in item:
                raise RuntimeError(str(item["error"]))
            if item.get("usage") is not None:
                usage = item["usage"]
            for choice in item.get("choices", []):
                delta = choice.get("delta", {})
                if delta.get("content") or delta.get("reasoning_content") or delta.get("tool_calls"):
                    times.append(time.perf_counter() - started)
                content.append(delta.get("content") or "")
                reasoning.append(delta.get("reasoning_content") or "")
                # Generated call IDs are intentionally excluded from equality.
                calls.extend({k: v for k, v in call.items() if k != "id"}
                             for call in delta.get("tool_calls", []))
                finish = choice.get("finish_reason") or finish
    if not done or usage is None or finish is None:
        raise ValueError("incomplete stream: expected usage, finish_reason and [DONE]")
    elapsed = time.perf_counter() - started
    generation_s = elapsed - times[0] if times else None
    output_tokens = usage.get("completion_tokens", 0)
    # SSE chunks can contain multiple tokens; this is an estimate, not engine timing.
    generation_rate = ((output_tokens - 1) / generation_s
                       if len(times) > 1 and output_tokens > 1 and generation_s > 0 else None)
    gaps = [b - a for a, b in zip(times, times[1:])]
    output = {"content": "".join(content), "reasoning_content": "".join(reasoning),
              "tool_calls": calls, "finish_reason": finish}
    digest = hashlib.sha256(json.dumps(output, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return {"case": case["name"], "cache_enabled": cache, "elapsed_s": elapsed,
            "first_content_s": times[0] if times else None,
            "generation_tokens_s_estimate": generation_rate,
            "content_chunk_gaps_s": gaps, "chunk_gap_p95_s": percentile(gaps, .95),
            "chunk_gap_p99_s": percentile(gaps, .99), "usage": usage,
            "output": output, "output_sha256": digest}


class MetricsSampler:
    def __init__(self, url):
        self.url = url
        self.stop = threading.Event()
        self.peak = {}
        self.last = None
        self.error = None
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        while not self.stop.is_set():
            try:
                metrics = get_json(self.url + "/api/stats", timeout=2)
                for device, used in metrics.get("gpus", {}).get("memory_used_gb", {}).items():
                    self.peak[device] = max(self.peak.get(device, 0), used)
                self.last = {key: metrics.get(key) for key in ("last_prefill", "mtp", "active_decode_slots")}
                self.last["prefill_interleave"] = metrics.get("cache", {}).get("prefill_interleave")
            except Exception as exc:
                self.error = str(exc)
            self.stop.wait(.5)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(5)


def default_cases(repeats=64, image=None):
    prefix = "\n".join(f"Record {i}: station cedar, code {1000 + i}, state ready." for i in range(repeats))
    system = {"role": "system", "content": "Answer the final question briefly and exactly."}
    def case(name, text):
        return {"name": name, "messages": [system, {"role": "user", "content": text}]}
    cases = [case("original", prefix + "\nWhat is the code of record 0?"),
             case("tail_edit", prefix + "\nWhat is the code of record 1?"),
             case("branch", prefix[:len(prefix) // 2] + "\nReturn only the word READY.")]
    if image:
        cases.append({"name": "late_image", "messages": [system, {"role": "user", "content": [
            {"type": "text", "text": prefix},
            {"type": "image_url", "image_url": {"url": image}},
            {"type": "text", "text": "Describe the image in one sentence."}]}]})
    return cases


def long_output_cases(repeats=128):
    case = default_cases(repeats)[0]
    case["name"] = "long_output"
    case["messages"][0]["content"] = "Follow the final instruction exactly."
    case["messages"][-1]["content"] = case["messages"][-1]["content"].rsplit("\n", 1)[0] + (
        "\nOutput all integers from 1 through 1000, separated by commas. "
        "Do not skip any numbers. No explanation.")
    return [case]


def run_suite(url, cases, *, concurrency=3, max_tokens=32, timeout=300):
    url = url.rstrip("/")
    model = get_json(url + "/v1/models")["data"][0]["id"]
    report = {"url": url, "model": model, "concurrency": concurrency,
              "max_tokens": max_tokens, "phases": {}, "errors": []}
    references = {}
    for phase, cache, workers in (("cold", False, 1), ("warmup", True, 1),
                                   ("cached", True, 1), ("concurrent", True, concurrency),
                                   ("concurrent_cold", False, concurrency)):
        jobs = cases if not phase.startswith("concurrent") else [case for case in cases for _ in range(concurrency)]
        def execute(case):
            try:
                result = run_request(url, model, case, cache, max_tokens, timeout)
                if workers == 1:
                    try:
                        result["reported_prefill"] = get_json(url + "/api/stats").get("cache", {}).get("last_prefill")
                    except Exception as exc:
                        result["metrics_error"] = str(exc)
                return result
            except Exception as exc:
                return {"case": case["name"], "error": str(exc)}
        started = time.perf_counter()
        with MetricsSampler(url) as sampler:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(execute, jobs))
            elapsed = time.perf_counter() - started
        for result in results:
            if "error" in result:
                report["errors"].append({"phase": phase, **result})
                continue
            if phase == "cold":
                references[result["case"]] = result["output_sha256"]
            result["matches_cold"] = result["output_sha256"] == references.get(result["case"])
        gaps = [gap for r in results for gap in r.get("content_chunk_gaps_s", [])]
        first = [r["first_content_s"] for r in results if r.get("first_content_s") is not None]
        rates = [r["generation_tokens_s_estimate"] for r in results
                 if r.get("generation_tokens_s_estimate") is not None]
        report["phases"][phase] = {
            "results": results, "wall_s": elapsed,
            "aggregate_output_tokens_s": sum(r.get("usage", {}).get("completion_tokens", 0) for r in results) / max(elapsed, 1e-9),
            "first_content_p95_s": percentile(first, .95),
            "generation_tokens_s_estimate_median": percentile(rates, .5),
            "chunk_gap_p95_s": percentile(gaps, .95), "chunk_gap_p99_s": percentile(gaps, .99),
            "sampled_peak_gpu_memory_gb": sampler.peak, "last_metrics": sampler.last,
            "metrics_error": sampler.error,
        }
        print(f"{phase}: {len(results)} requests in {elapsed:.2f}s", flush=True)
        if rates:
            total_rate = report["phases"][phase]["aggregate_output_tokens_s"]
            print(f"  近似生成速率中位数 {percentile(rates, .5):.2f} tok/s/请求；"
                  f"含排队及预填充的总吞吐 {total_rate:.2f} tok/s", flush=True)
    report["consistent"] = not report["errors"] and all(
        r.get("matches_cold", False) for p in report["phases"].values() for r in p["results"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="API origin without /v1")
    parser.add_argument("--label", default="current", help="e.g. mtp-on or mtp-off")
    parser.add_argument("--cases", type=Path, help="JSON array of {name, messages}")
    parser.add_argument("--long-output", action="store_true", help="long generation speed test (default 512 output tokens)")
    parser.add_argument("--image", help="optional image URL/data URL or server-local path")
    parser.add_argument("--prefix-records", type=int, default=64)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, help="output limit (default 32, or 512 with --long-output)")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--compare", type=Path, help="compare cold output with an earlier report")
    args = parser.parse_args()
    if args.long_output and (args.cases or args.image):
        parser.error("--long-output cannot be combined with --cases or --image")
    if args.max_tokens is None:
        args.max_tokens = 512 if args.long_output else 32
    if min(args.concurrency, args.max_tokens, args.prefix_records, args.timeout) <= 0:
        parser.error("counts and timeout must be positive")
    cases = (json.loads(args.cases.read_text()) if args.cases else
             long_output_cases(args.prefix_records) if args.long_output else
             default_cases(args.prefix_records, args.image))
    if (not isinstance(cases, list) or not cases or any(not isinstance(c, dict) or not isinstance(c.get("name"), str)
            or not isinstance(c.get("messages"), list) or not c["messages"] for c in cases)
            or len({c["name"] for c in cases}) != len(cases)):
        parser.error("cases must be non-empty and have distinct names and message arrays")
    report = run_suite(args.url, cases, concurrency=args.concurrency, max_tokens=args.max_tokens, timeout=args.timeout)
    report.update(label=args.label, python=platform.python_version(), cases=cases,
                  measurement="First content arrival and SSE content-chunk gaps; not per-token latency. GPU peaks sampled at 0.5s.")
    if args.compare:
        reference = json.loads(args.compare.read_text())
        comparable = (reference["model"] == report["model"] and reference["cases"] == cases
                      and reference["max_tokens"] == args.max_tokens)
        previous = {r["case"]: r.get("output_sha256") for r in reference["phases"]["cold"]["results"]}
        report["comparison"] = {"reference": str(args.compare), "comparable": comparable,
                                "matches": comparable and not reference.get("errors") and not report["errors"]
                                and all(r.get("output_sha256") == previous.get(r["case"])
                                    for r in report["phases"]["cold"]["results"])}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"Report: {args.report}; consistent={report['consistent']}")
    if not report["consistent"] or not report.get("comparison", {"matches": True})["matches"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
