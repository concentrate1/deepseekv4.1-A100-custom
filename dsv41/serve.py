"""OpenAI-compatible HTTP server (no web framework needed: stdlib http.server, threaded).

  python -m dsv41.serve --devices 0,1,2,3,4 --port 8000
  curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
       -d '{"model":"deepseek-v4.1-flash","messages":[{"role":"user","content":"hello"}],"stream":true}'

Endpoints: GET /v1/models, POST /v1/chat/completions (stream or not), POST /v1/completions, GET /health, GET /dashboard.
Generation is serialized or batched according to the worker profile."""
import sys
import torch
import argparse
import json
import time
import os
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import copy

import os as _os
import traceback
import threading
import math
import select
import socket
_os.environ.setdefault("OMP_WAIT_POLICY", "active")  # CPU expert threads keep spinning between layers (libgomp reads this once)
_os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from .engine import Engine, GenParams, parse_budgets
from .stats import StatsTracker
from .streaming import ChatStreamSplitter
from .sse import SSEWriter

ENGINE: Engine | None = None
STATS_TRACKER: StatsTracker | None = None



def _reasoning_mode_and_effort(body: dict) -> tuple[str | None, str | int | None]:
    """Map API effort to the model encoder's 1-100 scale.

    The checkpoint defines low=50, high=75, max=100. Medium and the
    additional API labels below are explicit local interpolations.
    """
    effort = body.get("reasoning_effort")
    if effort is None:
        return ("thinking", None) if body.get("thinking") else (None, None)
    if type(effort) is int and 1 <= effort <= 100:
        return "thinking", effort
    if isinstance(effort, str):
        if effort == "none":
            return None, None
        labels = {"minimal": 1, "low": "low", "medium": 65,
                  "high": "high", "xhigh": 90, "max": "max"}
        if effort in labels:
            return "thinking", labels[effort]
    raise ValueError("reasoning_effort must be an integer 1-100 or one of "
                     "none, minimal, low, medium, high, xhigh, max")


def _params(body: dict) -> GenParams:
    stop = body.get("stop") or []
    if isinstance(stop, str):
        stop = [stop]
    default_rep_pen = float(os.environ.get("DSV41_REPETITION_PENALTY", "1.0"))
    default_pres_pen = float(os.environ.get("DSV41_PRESENCE_PENALTY", "0.0"))
    default_freq_pen = float(os.environ.get("DSV41_FREQUENCY_PENALTY", "0.0"))
    default_window = int(os.environ.get("DSV41_PENALTY_WINDOW", "256"))
    default_prog_pen = float(os.environ.get("DSV41_PROGRESSIVE_PENALTY", "0.0"))
    default_ban_cycles = os.environ.get("DSV41_BAN_CYCLES", "0").strip().lower() not in ("0", "false", "off")
    default_loop_detect = os.environ.get("DSV41_LOOP_DETECT", "0").strip().lower() not in ("0", "false", "off")
    default_min_loop_match = int(os.environ.get("DSV41_MIN_LOOP_MATCH", "48"))
    default_min_loop_cycle = int(os.environ.get("DSV41_MIN_LOOP_CYCLE", "1"))

    rep_pen = float(body.get("repetition_penalty") if body.get("repetition_penalty") is not None else default_rep_pen)
    pres_pen = float(body.get("presence_penalty") if body.get("presence_penalty") is not None else default_pres_pen)
    freq_pen = float(body.get("frequency_penalty") if body.get("frequency_penalty") is not None else default_freq_pen)
    window = int(body.get("penalty_window") or default_window)
    prog_pen = float(body.get("progressive_penalty") if body.get("progressive_penalty") is not None else default_prog_pen)
    ban_cycles = bool(body.get("ban_cycles") if body.get("ban_cycles") is not None else default_ban_cycles)
    loop_detect = bool(body.get("loop_detect") if body.get("loop_detect") is not None else default_loop_detect)
    min_loop_match = int(body.get("min_loop_match") or default_min_loop_match)
    min_loop_cycle = int(body.get("min_loop_cycle") or default_min_loop_cycle)

    return GenParams(
        max_new_tokens=int(body.get("max_tokens") or body.get("max_completion_tokens") or int(os.environ.get("DSV41_INTERACTIVE_MAX_NEW", "65536"))),
        temperature=float(body.get("temperature", 1.0)),
        top_p=float(body.get("top_p", 0.95)),
        stop=list(stop),
        seed=body.get("seed"),
        repetition_penalty=rep_pen,
        presence_penalty=pres_pen,
        frequency_penalty=freq_pen,
        penalty_window=window,
        progressive_penalty=prog_pen,
        ban_cycles=ban_cycles,
        loop_detect=loop_detect,
        min_loop_match=min_loop_match,
        min_loop_cycle=min_loop_cycle,
    )


def validate_request(body):
    if not isinstance(body, dict):
        raise ValueError("request body must be a JSON object")
    rf = body.get("response_format")
    if rf is None:
        body["response_format"] = {}
    elif not isinstance(rf, dict):
        raise ValueError("response_format must be an object or null")
    schema = body["response_format"].get("json_schema")
    if schema is not None and not isinstance(schema, dict):
        raise ValueError("response_format.json_schema must be an object")
    if schema is None:
        body["response_format"]["json_schema"] = {}
    if isinstance(schema, dict) and schema.get("schema") is not None and not isinstance(schema["schema"], dict):
        raise ValueError("response_format.json_schema.schema must be an object")
    messages = body.get("messages")
    if messages is not None:
        if not isinstance(messages, list) or any(not isinstance(m, dict) for m in messages):
            raise ValueError("messages must be an array of objects")
        if any(m.get("role") not in ("system", "developer", "user", "assistant", "tool") for m in messages):
            raise ValueError("messages contain an invalid role")
    if "prompt" in body:
        prompt = body["prompt"]
        if not isinstance(prompt, str) and not (isinstance(prompt, list) and prompt and all(isinstance(x, str) for x in prompt)):
            raise ValueError("prompt must be a string or a non-empty array of strings; token IDs are unsupported")
    for name in ("max_tokens", "max_completion_tokens", "penalty_window", "min_loop_match", "min_loop_cycle", "max_batch"):
        value = body.get(name)
        if value is not None and (type(value) is not int or value <= 0):
            raise ValueError(f"{name} must be a positive integer")
    for name in ("temperature", "top_p", "repetition_penalty", "frequency_penalty", "presence_penalty", "progressive_penalty"):
        value = body.get(name)
        if name in body and (type(value) not in (int, float) or not math.isfinite(value)):
            raise ValueError(f"{name} must be a finite number")
    if body.get("temperature", 1) < 0 or not 0 < body.get("top_p", .95) <= 1:
        raise ValueError("temperature must be non-negative and top_p must be in (0, 1]")
    if body.get("repetition_penalty", 1) <= 0:
        raise ValueError("repetition_penalty must be positive")
    if body.get("seed") is not None and type(body["seed"]) is not int:
        raise ValueError("seed must be an integer")
    stop = body.get("stop")
    if stop is not None and not isinstance(stop, str) and not (isinstance(stop, list) and all(isinstance(x, str) for x in stop)):
        raise ValueError("stop must be a string or array of strings")
    for name in ("stream", "thinking", "jev", "ban_cycles", "loop_detect"):
        if name in body and type(body[name]) is not bool:
            raise ValueError(f"{name} must be boolean")
    tools = body.get("tools")
    if tools is not None and (not isinstance(tools, list) or any(not isinstance(t, dict) for t in tools)):
        raise ValueError("tools must be an array of objects")
    options = body.get("stream_options")
    if options is not None and (not isinstance(options, dict) or ("include_usage" in options and type(options["include_usage"]) is not bool)):
        raise ValueError("stream_options must be an object with boolean include_usage")
    if body.get("schema") is not None and not isinstance(body["schema"], dict):
        raise ValueError("schema must be an object")
    for message in messages or []:
        content = message.get("content")
        if content is not None and not isinstance(content, (str, list)):
            raise ValueError("message content must be a string, array or null")
        if isinstance(content, list) and any(not isinstance(part, dict) for part in content):
            raise ValueError("message content parts must be objects")
    _reasoning_mode_and_effort(body)
    _params(body)
    return body


_MTP_LIMIT = threading.BoundedSemaphore(8)
_MTP_LOCK = threading.Lock()
_MTP_CACHE = (None, 0.0, {})


def shared_mtp_stats():
    global _MTP_CACHE
    with _MTP_LOCK:
        owner, timestamp, value = _MTP_CACHE
        now = time.monotonic()
        if owner is not ENGINE or now - timestamp >= .3:
            value = ENGINE.get_cache_stats().get("mtp", {})
            _MTP_CACHE = (ENGINE, time.monotonic(), value)
        return value


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quieter log
        if getattr(self, "path", "") in ("/api/metrics", "/api/stats", "/health", "/favicon.ico"):
            return
        print(f"[{time.strftime('%H:%M:%S')}] {self.address_string()} {fmt % args}", flush=True)

    def _json(self, code: int, obj: dict, t0: float | None = None, is_stream: bool = False):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        try:
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            elapsed_str = f" elapsed={time.perf_counter()-t0:.3f}s" if t0 is not None else ""
            print(f"[http] client disconnected before response completed{elapsed_str} bytes={len(data)} stream={is_stream}", flush=True)
            if STATS_TRACKER:
                STATS_TRACKER.record_disconnect()
        except Exception as e:
            print(f"[http] response write error: {e}", flush=True)

    def do_GET(self):
        if self.path == "/v1/models":
            model = ENGINE.model_name if ENGINE else os.environ.get("DSV41_MODEL_NAME", "deepseek-v4.1-flash")
            self._json(200, {"object": "list", "data": [
                {"id": model, "object": "model", "owned_by": "local"}
            ]})
        elif self.path == "/health":
            if hasattr(ENGINE, "worker_health"):
                try:
                    self._json(200, ENGINE.worker_health())
                except Exception as exc:
                    self._json(503, {"status": "unavailable", "error": str(exc)})
            else:
                self._json(200, {"status": "ok"})
        elif self.path in ("/dashboard", "/"):
            self._dashboard()
        elif self.path == "/api/mtp/stream":
            self._mtp_stream()
        elif self.path in ("/api/metrics", "/api/stats"):
            self._json(200, STATS_TRACKER.get_metrics(ENGINE) if STATS_TRACKER else {})
        else:
            self._json(404, {"error": "not found"})

    def _mtp_stream(self):
        if not _MTP_LIMIT.acquire(blocking=False):
            return self._json(429, {"error": "too many MTP stream connections"})
        try:
            self._mtp_stream_body()
        finally:
            _MTP_LIMIT.release()
            self.close_connection = True

    def _mtp_stream_body(self):
        """Push shared counters rather than query the worker per connection."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        previous = None
        last_heartbeat = time.monotonic()
        try:
            while True:
                mtp = shared_mtp_stats()
                encoded = json.dumps(mtp, ensure_ascii=False, sort_keys=True)
                if encoded != previous:
                    self.wfile.write(("data: " + encoded + "\n\n").encode("utf-8"))
                    self.wfile.flush()
                    previous = encoded
                elif time.monotonic() - last_heartbeat >= 10:
                    self.wfile.write(b": keep-alive\n\n")
                    self.wfile.flush()
                    last_heartbeat = time.monotonic()
                time.sleep(0.3)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        except Exception as exc:
            print(f"[mtp-stream] stopped: {exc}", flush=True)
        finally:
            self.close_connection = True

    def _dashboard(self):
        model_name = ENGINE.model_name if ENGINE else "deepseek-v4.1-flash"
        try:
            if STATS_TRACKER:
                html = STATS_TRACKER.render_dashboard_html(model_name).encode("utf-8")
            else:
                from .stats import _DASHBOARD_HTML_TEMPLATE
                html = _DASHBOARD_HTML_TEMPLATE.replace("__MODEL_NAME__", model_name).encode("utf-8")
        except Exception as e:
            html = f"<html><body><h1>Dashboard Error</h1><p>{e}</p></body></html>".encode("utf-8")
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _watch_client(self, stop, cancel):
        """Observe TCP closure without consuming request or pipelined bytes."""
        while not stop.is_set():
            try:
                readable, _, _ = select.select([self.connection], [], [], .2)
                if readable:
                    if self.connection.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT) == b"":
                        cancel.set()
                        return
                    stop.wait(.2)
            except BlockingIOError:
                continue
            except OSError:
                cancel.set()
                return

    def _cancel_kwargs(self):
        return {"cancel_event": self._request_cancel} if hasattr(self, "_request_cancel") else {}

    def do_POST(self):
        watch_stop = threading.Event()
        watcher = None
        if STATS_TRACKER:
            STATS_TRACKER.client_connected()
        try:
            try:
                n = int(self.headers.get("Content-Length", 0))
                if n < 0:
                    raise ValueError("Content-Length must be non-negative")
                body = validate_request(json.loads(self.rfile.read(n) or b"{}"))
            except (ValueError, TypeError, OverflowError) as exc:
                self.close_connection = True
                return self._json(400, {"error": {"message": str(exc), "type": "invalid_request_error"}})
            if hasattr(self, "connection"):
                self._request_cancel = threading.Event()
                watcher = threading.Thread(target=self._watch_client, args=(watch_stop, self._request_cancel), daemon=True)
                watcher.start()
            if self.path in ("/v1/chat/completions", "/v1/completions"):
                requested_model = body.get("model")
                if requested_model and requested_model != ENGINE.model_name:
                    return self._json(404, {"error": {
                        "message": f"Model {requested_model!r} is not available; loaded model is {ENGINE.model_name!r}",
                        "type": "invalid_request_error", "code": "model_not_found",
                    }})
                rf_type = body.get("response_format", {}).get("type")
                has_rf_schema = rf_type == "json_schema" and "schema" in body.get("response_format", {}).get("json_schema", {})
                if body.get("jev") or body.get("mode") == "jev" or has_rf_schema or body.get("schema"):
                    return self._jev(body)
            if self.path == "/v1/chat/completions":
                return self._chat(body)
            if self.path == "/v1/completions":
                return self._completion(body)
            self._json(404, {"error": "not found"})
        finally:
            watch_stop.set()
            if watcher is not None:
                watcher.join(timeout=1)
            if STATS_TRACKER:
                STATS_TRACKER.client_disconnected()


    # ---------------------------------------------------------------- chat
    def _chat(self, body: dict):
        eng = ENGINE

        # Per-request prefix cache control:
        #
        #   X-DSV41-Prefix-Cache: off
        #   X-DSV41-Prefix-Cache: on
        #
        # "off" bypasses prefix reuse only for this request.
        # "on" leaves normal cache behaviour enabled.
        _prefix_header = (
            self.headers.get(
                "X-DSV41-Prefix-Cache",
                "",
            )
            .strip()
            .lower()
        )

        if _prefix_header in ("off", "0", "false", "disable", "disabled"):
            eng._disable_prefix_cache_once = True
            print(
                "[http-debug] prefix-cache=OFF for this request",
                flush=True,
            )
        elif _prefix_header in ("on", "1", "true", "enable", "enabled"):
            eng._disable_prefix_cache_once = False
            print(
                "[http-debug] prefix-cache=ON for this request",
                flush=True,
            )
        elif hasattr(eng, "socket_path"):
            eng._disable_prefix_cache_once = False
        messages = copy.deepcopy(body.get("messages") or [])
        # A client may provide tool schemas but explicitly forbid calls for
        # this turn. Keep those schemas out of the model prompt in that case.
        tools = None if body.get("tool_choice") == "none" else body.get("tools")
        if tools and messages:
            if messages[0].get("role") == "system":
                messages[0]["tools"] = tools
            else:
                messages.insert(0, {"role": "system", "content": "", "tools": tools})
        try:
            thinking, reasoning_effort = _reasoning_mode_and_effort(body)
            ids, images, token_types = eng.format_chat(
                messages, thinking, reasoning_effort=reasoning_effort)
        except Exception as e:  # malformed messages / unsupported content
            return self._json(400, {"error": f"cannot encode messages: {e}"})
        params = _params(body)
        # Claude/LiteLLM may retry a stalled stream as a non-stream request
        # with max_tokens=32000.  Keep interactive retries bounded so a
        # response reaches the client and the request can complete.  Hosts
        # that need longer answers can raise this explicitly.
        _interactive_cap = int(os.environ.get("DSV41_INTERACTIVE_MAX_NEW", "65536"))
        requested_max_tokens = body.get("max_tokens") or body.get("max_completion_tokens")
        if _interactive_cap > 0 and params.max_new_tokens > _interactive_cap:
            params.max_new_tokens = _interactive_cap
            print(f"[chat] max_tokens capped={_interactive_cap} (requested={requested_max_tokens})", flush=True)
        rid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        if STATS_TRACKER:
            STATS_TRACKER.record_input(body, "chat/completions", len(ids), rid)
        created = int(time.time())
        t0 = time.perf_counter()
        print(
          f"[chat] START req_id={rid} prompt_tokens={len(ids)} "
          f"max_seq_len={eng.max_seq_len} "
          f"max_tokens={params.max_new_tokens} (effective) "
          f"stream={body.get('stream')} "
          f"temperature={params.temperature} top_p={params.top_p} "
          f"rep_pen={params.repetition_penalty} freq_pen={params.frequency_penalty} "
          f"prog_pen={params.progressive_penalty} ban_cycles={params.ban_cycles} "
          f"loop_detect={params.loop_detect} penalty_window={params.penalty_window}",
          flush=True,
        )
        if body.get("stream"):
            # Send SSE headers immediately and emit keep-alive comments while
            # long-context prefill/generation is running. Previously the
            # server buffered the entire response, so LiteLLM timed out on
            # 100K+ prompts despite stream=True.
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            writer = SSEWriter(self.wfile, getattr(self, "_request_cancel", None)).start()
            _sse_write = writer.write
            if STATS_TRACKER:
                STATS_TRACKER.record_request_start(len(ids), stream=True)
            t_gen_0 = time.perf_counter()
            n = 0
            raw = ""
            splitter = ChatStreamSplitter(thinking=bool(thinking))
            def chunk(delta, finish=None, usage=None):
                obj = {"id": rid, "object": "chat.completion.chunk", "created": created,
                       "model": body.get("model") or eng.model_name,
                       "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
                if usage is not None:
                    obj["usage"] = usage
                _sse_write(("data: " + json.dumps(obj, ensure_ascii=False) + "\n\n").encode("utf-8"))

            stream_iterator = None
            try:
                stream_iterator = eng.stream_text(ids, params, images=images, token_types=token_types, **self._cancel_kwargs())
                chunk({"role": "assistant"})
                for piece, final_count in stream_iterator:
                    if final_count is None:
                        raw += piece
                        for delta in splitter.push(piece):
                            chunk(delta)
                    else:
                        raw = piece
                        n = final_count
                msg = eng.parse_completion(raw, thinking)
                if not isinstance(msg, dict):
                    msg = {"content": raw, "tool_calls": []}
                reasoning = msg.get("reasoning_content") or ""
                content = msg.get("content") or ""
                tool_calls = msg.get("tool_calls") or []
                if not content and not tool_calls and not reasoning and not thinking:
                    content = raw
                if not reasoning.startswith(splitter.reasoning) or not content.startswith(splitter.content):
                    print(f"[stream-parse] final parse diverged req_id={rid}; already sent text cannot be replaced", flush=True)
                if reasoning.startswith(splitter.reasoning) and len(reasoning) > len(splitter.reasoning):
                    chunk({"reasoning_content": reasoning[len(splitter.reasoning):]})
                if content.startswith(splitter.content) and len(content) > len(splitter.content):
                    chunk({"content": content[len(splitter.content):]})
                for i, tc in enumerate(tool_calls):
                    fn = tc.get("function") or {}
                    args = fn.get("arguments", "")
                    if not isinstance(args, str):
                        args = json.dumps(args, ensure_ascii=False)
                    chunk({"tool_calls": [{"index": i, "id": tc.get("id") or f"call_{uuid.uuid4().hex[:24]}",
                                           "type": tc.get("type", "function"),
                                           "function": {"name": fn.get("name", ""), "arguments": args}}]})
                finish = "tool_calls" if tool_calls else ("length" if getattr(eng, "last_finish_reason", None) == "length" or n >= params.max_new_tokens else "stop")
                usage = {"prompt_tokens": len(ids), "completion_tokens": n,
                         "total_tokens": len(ids) + n}
                include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
                chunk({}, finish, None if include_usage else usage)
                if include_usage:
                    usage_chunk = {"id": rid, "object": "chat.completion.chunk", "created": created,
                                   "model": body.get("model") or eng.model_name,
                                   "choices": [], "usage": usage}
                    _sse_write(("data: " + json.dumps(usage_chunk, ensure_ascii=False) + "\n\n").encode("utf-8"))
                writer.finish()
            except (BrokenPipeError, ConnectionResetError, OSError):
                if STATS_TRACKER:
                    STATS_TRACKER.record_disconnect()
                return
            except Exception as e:
                tb = traceback.format_exc()
                print(f"[chat-error] stream failed req_id={rid}: {e}\n{tb}", flush=True)
                try:
                    _sse_write(("data: " + json.dumps({"error": {"message": str(e), "type": "generation_error"}}, ensure_ascii=False) + "\n\n").encode())
                    writer.finish()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
                return
            finally:
                writer.close()
                getattr(stream_iterator, "close", lambda: None)()
                self.close_connection = True
            dt_gen = time.perf_counter() - t_gen_0
            decode_tok_s = getattr(eng, "last_decode_tok_s", None)
            if decode_tok_s is None:
                decode_tok_s = n / max(dt_gen, 1e-6)
            print(f"[chat] END req_id={rid} tokens={n} time={dt_gen:.2f}s ({decode_tok_s:.1f} tok/s decode)", flush=True)
            if STATS_TRACKER and n > 0:
                STATS_TRACKER.record_throughput(decode_tok_s, "chat-stream")
                STATS_TRACKER.record_cache_event(f"Chat stream completed: {n} tokens in {dt_gen:.2f}s ({decode_tok_s:.1f} tok/s decode)")
            return
        if STATS_TRACKER:
            STATS_TRACKER.record_request_start(len(ids), stream=False)
        t_gen_0 = time.perf_counter()
        try:
            text, n = eng.generate_text(ids, params, images=images, token_types=token_types, **self._cancel_kwargs())
        except Exception as e:
            tb = traceback.format_exc()
            print(f"\n[chat-error] non-streaming generation failed req_id={rid}: {e}\n{tb}", flush=True)
            return self._json(500, {"error": {"message": str(e), "type": "server_error", "traceback": tb}}, t0=t0, is_stream=False)
        dt_gen = time.perf_counter() - t_gen_0
        decode_tok_s = getattr(eng, "last_decode_tok_s", None)
        if decode_tok_s is None:
            decode_tok_s = n / max(dt_gen, 1e-6)
        print(f"[chat] END req_id={rid} tokens={n} time={dt_gen:.2f}s ({decode_tok_s:.1f} tok/s decode)", flush=True)
        if STATS_TRACKER and n > 0:
            STATS_TRACKER.record_throughput(decode_tok_s, "chat")
            STATS_TRACKER.record_cache_event(
                f"Chat completed: {n} tokens in {dt_gen:.2f}s ({decode_tok_s:.1f} tok/s decode)"
            )
        msg = eng.parse_completion(text, thinking)
        if isinstance(msg, dict):
            for tc in msg.get("tool_calls") or []:
                tc.setdefault("id", f"call_{uuid.uuid4().hex[:24]}")
                tc.setdefault("type", "function")
        content = msg.get("content") if isinstance(msg, dict) else text
        out = {"id": rid, "object": "chat.completion", "created": created, "model": body.get("model") or eng.model_name,
               "choices": [{"index": 0, "message": {"role": "assistant", "content": content if content is not None else text},
                            "finish_reason": "tool_calls" if (isinstance(msg, dict) and msg.get("tool_calls")) else ("length" if (getattr(eng, "last_finish_reason", None) == "length" or n >= params.max_new_tokens) else "stop")}],
               "usage": {"prompt_tokens": len(ids), "completion_tokens": n, "total_tokens": len(ids) + n}}
        if isinstance(msg, dict) and msg.get("reasoning_content"):
            out["choices"][0]["message"]["reasoning_content"] = msg["reasoning_content"]
        if isinstance(msg, dict) and msg.get("tool_calls"):
            out["choices"][0]["message"]["tool_calls"] = msg["tool_calls"]
        self._json(200, out, t0=t0, is_stream=False)

    # ------------------------------------------------ raw completions
    def _completion(self, body: dict):
        eng = ENGINE
        t0 = time.perf_counter()
        prompt = body.get("prompt") or ""
        if isinstance(prompt, list):
            prompt = prompt[0]
        try:
            if "<image>" in prompt and "</image>" in prompt:
                ids, images, token_types = eng.format_chat([{"role": "user", "content": prompt}], thinking_mode="chat")
            else:
                ids = eng.tok.encode(prompt)
                images, token_types = None, None
        except (ValueError, TypeError) as exc:
            return self._json(400, {"error": str(exc)})
        params = _params(body)
        rid = f"cmpl-{uuid.uuid4().hex[:24]}"
        if STATS_TRACKER:
            STATS_TRACKER.record_input(body, "completions", len(ids), rid)
        created = int(time.time())
        if body.get("stream"):
            if STATS_TRACKER:
                STATS_TRACKER.record_request_start(len(ids), stream=True)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.close_connection = True
            self.end_headers()
            n = 0
            t_gen_0 = time.perf_counter()
            writer = SSEWriter(self.wfile, getattr(self, "_request_cancel", None)).start()
            stream_iterator = None
            try:
                stream_iterator = eng.stream_text(ids, params, images=images, token_types=token_types, **self._cancel_kwargs())
                emitted = ""
                for piece, final_count in stream_iterator:
                    if final_count is not None:
                        n = final_count
                        if not piece.startswith(emitted):
                            print(f"[stream-parse] completion diverged req_id={rid}", flush=True)
                        piece = piece[len(emitted):] if piece.startswith(emitted) else ""
                    else:
                        emitted += piece
                    if not piece:
                        continue
                    obj = {"id": rid, "object": "text_completion", "created": created, "model": body.get("model") or eng.model_name,
                           "choices": [{"index": 0, "text": piece, "finish_reason": None}]}
                    writer.write(f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode())
                usage = {"prompt_tokens": len(ids), "completion_tokens": n,
                         "total_tokens": len(ids) + n}
                finish = "length" if getattr(eng, "last_finish_reason", None) == "length" or n >= params.max_new_tokens else "stop"
                include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
                finish_chunk = {"id": rid, "object": "text_completion", "created": created,
                                "model": body.get("model") or eng.model_name,
                                "choices": [{"index": 0, "text": "", "finish_reason": finish}]}
                if not include_usage:
                    finish_chunk["usage"] = usage
                writer.write(("data: " + json.dumps(finish_chunk, ensure_ascii=False) + "\n\n").encode())
                if include_usage:
                    usage_chunk = {"id": rid, "object": "text_completion", "created": created,
                                   "model": body.get("model") or eng.model_name,
                                   "choices": [], "usage": usage}
                    writer.write(("data: " + json.dumps(usage_chunk, ensure_ascii=False) + "\n\n").encode())
                writer.finish()
                dt_gen = time.perf_counter() - t_gen_0
                decode_tok_s = getattr(eng, "last_decode_tok_s", None)
                if decode_tok_s is None:
                    decode_tok_s = n / max(dt_gen, 1e-6)
                if STATS_TRACKER and n > 0:
                    STATS_TRACKER.record_throughput(decode_tok_s, "completion-stream")
                    STATS_TRACKER.record_cache_event(
                        f"Completion stream: {n} tokens in {dt_gen:.2f}s ({decode_tok_s:.1f} tok/s decode)"
                    )
            except (BrokenPipeError, ConnectionResetError):
                elapsed = time.perf_counter() - t0
                print(f"[http] client disconnected during completion stream elapsed={elapsed:.3f}s tokens={n}", flush=True)
                if STATS_TRACKER:
                    STATS_TRACKER.record_disconnect()
            except Exception as exc:
                print(f"[completion-error] stream failed: {exc}", flush=True)
                try:
                    writer.write(("data: " + json.dumps({"error": {"message": str(exc), "type": "generation_error"}}) + "\n\n").encode())
                    writer.finish()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
            finally:
                writer.close()
                getattr(stream_iterator, "close", lambda: None)()
                self.close_connection = True
            return
        if STATS_TRACKER:
            STATS_TRACKER.record_request_start(len(ids), stream=False)
        t_gen_0 = time.perf_counter()
        try:
            text, n = eng.generate_text(ids, params, images=images, token_types=token_types, **self._cancel_kwargs())
        except Exception as e:
            tb = traceback.format_exc()
            print(f"\n[completion-error] generation failed req_id={rid}: {e}\n{tb}", flush=True)
            return self._json(500, {"error": {"message": str(e), "type": "server_error", "traceback": tb}}, t0=t0, is_stream=False)
        dt_gen = time.perf_counter() - t_gen_0
        decode_tok_s = getattr(eng, "last_decode_tok_s", None)
        if decode_tok_s is None:
            decode_tok_s = n / max(dt_gen, 1e-6)
        if STATS_TRACKER and n > 0:
            STATS_TRACKER.record_throughput(decode_tok_s, "completion")
            STATS_TRACKER.record_cache_event(
                f"Completion: {n} tokens in {dt_gen:.2f}s ({decode_tok_s:.1f} tok/s decode)"
            )
        self._json(200, {"id": rid, "object": "text_completion", "created": created, "model": body.get("model") or eng.model_name,
                          "choices": [{"index": 0, "text": text, "finish_reason": "length" if (getattr(eng, "last_finish_reason", None) == "length" or n >= params.max_new_tokens) else "stop"}],
                          "usage": {"prompt_tokens": len(ids), "completion_tokens": n, "total_tokens": len(ids) + n}}, t0=t0, is_stream=False)

    # ---------------------------------------------------------------- Jev mode structured output
    def _jev(self, body: dict):
        eng = ENGINE
        t0 = time.perf_counter()
        stream = bool(body.get("stream"))

        raw_schema = body.get("schema")
        if not raw_schema and isinstance(body.get("response_format"), dict):
            rf = body["response_format"]
            if rf.get("type") == "json_schema" and isinstance(rf.get("json_schema"), dict):
                raw_schema = rf["json_schema"].get("schema")
        if not raw_schema:
            return self._json(400, {"error": "schema is required for Jev mode"}, t0=t0, is_stream=stream)

        prompt = body.get("prompt")
        if not prompt and body.get("messages"):
            for m in reversed(body["messages"]):
                if m.get("role") == "user":
                    prompt = m.get("content")
                    break
            if not prompt:
                prompt = body["messages"][-1].get("content", "")

        if isinstance(prompt, list) and prompt and all(isinstance(x, str) for x in prompt):
            prompt = prompt[0]
        if not isinstance(prompt, str) or not prompt:
            return self._json(400, {"error": "a text prompt or user message is required"}, t0=t0, is_stream=stream)

        max_batch = int(body.get("max_batch") or os.environ.get("DSV41_JEV_MAX_BATCH", "32"))
        rid = f"jevcmpl-{uuid.uuid4().hex[:24]}"
        if STATS_TRACKER:
            STATS_TRACKER.record_input(body, "jev", 0, rid)
        created = int(time.time())

        print(
            f"[jev] request start prompt_len={len(prompt)} schema_fields={len(raw_schema)} stream={stream}",
            flush=True,
        )
        if STATS_TRACKER:
            STATS_TRACKER.record_request_start(len(prompt), stream=stream)

        if stream:
            # SSE streaming response with heartbeat to prevent client / proxy timeouts
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.close_connection = True
            self.end_headers()

            writer = SSEWriter(self.wfile, getattr(self, "_request_cancel", None)).start()
            _sse_write = writer.write

            try:
                assembled, metrics = eng.jev_inference(prompt, raw_schema, max_batch=max_batch)
            except Exception as e:
                tb = traceback.format_exc()
                print(f"[jev-error] {e}\n{tb}", flush=True)
                try:
                    err_payload = json.dumps({"error": {"message": str(e), "type": "jev_error"}}, ensure_ascii=False)
                    writer.finish(f"data: {err_payload}\n\ndata: [DONE]\n\n".encode("utf-8"))
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
                return
            finally:
                writer.close()

            json_content = json.dumps(assembled, ensure_ascii=False)
            dt_total = time.perf_counter() - t0
            eff_tokens = metrics.get("completion_tokens", 0)
            if not eff_tokens:
                eff_tokens = len(eng.tok.encode(json_content, add_special_tokens=False)) if getattr(eng, "tok", None) else max(1, len(json_content.split()))

            if STATS_TRACKER:
                STATS_TRACKER.record_throughput(eff_tokens / max(dt_total, 1e-6), "jev-stream")
                reused = metrics.get("prefix_saved_tokens", 0)
                tot = metrics.get("prompt_tokens", 0)
                hit_str = "HIT" if metrics.get("cache_hit") else "MISS"
                STATS_TRACKER.record_cache_event(
                    f"Jev stream [schema {hit_str}]: {metrics.get('num_fields', 0)} fields, {eff_tokens} output toks, saved {reused}/{tot} tokens ({dt_total*1000:.1f}ms)",
                    event_type="hit" if metrics.get("cache_hit") else "info"
                )

            try:
                role_chunk = {
                    "id": rid,
                    "object": "chat.completion.chunk" if self.path == "/v1/chat/completions" else "text_completion",
                    "created": created,
                    "model": body.get("model") or eng.model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant"},
                            "finish_reason": None,
                        }
                    ],
                }
                _sse_write(f"data: {json.dumps(role_chunk, ensure_ascii=False)}\n\n".encode("utf-8"))

                for i in range(0, len(json_content), 128):
                    content_chunk = {
                        "id": rid,
                        "object": "chat.completion.chunk" if self.path == "/v1/chat/completions" else "text_completion",
                        "created": created,
                        "model": body.get("model") or eng.model_name,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": json_content[i:i + 128]},
                                "finish_reason": None,
                            }
                        ],
                    }
                    _sse_write(f"data: {json.dumps(content_chunk, ensure_ascii=False)}\n\n".encode("utf-8"))

                finish_chunk = {
                    "id": rid,
                    "object": "chat.completion.chunk" if self.path == "/v1/chat/completions" else "text_completion",
                    "created": created,
                    "model": body.get("model") or eng.model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": metrics.get("prompt_tokens", 0),
                        "completion_tokens": eff_tokens,
                        "total_tokens": metrics.get("prompt_tokens", 0) + eff_tokens,
                    },
                    "jev_result": assembled,
                    "jev_metrics": metrics,
                }
                _sse_write(f"data: {json.dumps(finish_chunk, ensure_ascii=False)}\n\n".encode("utf-8"))
                _sse_write(b"data: [DONE]\n\n")
            except (BrokenPipeError, ConnectionResetError, OSError):
                elapsed = time.perf_counter() - t0
                print(
                    f"[http] broken pipe elapsed={elapsed:.3f}s bytes={len(json_content)} stream=True",
                    flush=True,
                )
                if STATS_TRACKER:
                    STATS_TRACKER.record_disconnect()
            return

        # Non-streaming response
        try:
            assembled, metrics = eng.jev_inference(prompt, raw_schema, max_batch=max_batch)
        except Exception as e:
            tb = traceback.format_exc()
            print(f"[jev-error] {e}\n{tb}", flush=True)
            return self._json(500, {"error": str(e), "traceback": tb}, t0=t0, is_stream=False)

        json_content = json.dumps(assembled, ensure_ascii=False)
        dt_total = time.perf_counter() - t0
        eff_tokens = metrics.get("completion_tokens", 0)
        if not eff_tokens:
            eff_tokens = len(eng.tok.encode(json_content, add_special_tokens=False)) if getattr(eng, "tok", None) else max(1, len(json_content.split()))

        if STATS_TRACKER:
            STATS_TRACKER.record_throughput(eff_tokens / max(dt_total, 1e-6), "jev")
            reused = metrics.get("prefix_saved_tokens", 0)
            tot = metrics.get("prompt_tokens", 0)
            hit_str = "HIT" if metrics.get("cache_hit") else "MISS"
            STATS_TRACKER.record_cache_event(
                f"Jev [schema {hit_str}]: {metrics.get('num_fields', 0)} fields, {eff_tokens} output toks, saved {reused}/{tot} tokens ({dt_total*1000:.1f}ms)",
                event_type="hit" if metrics.get("cache_hit") else "info"
            )

        resp = {
            "id": rid,
            "object": "chat.completion" if self.path == "/v1/chat/completions" else "text_completion",
            "created": created,
            "model": body.get("model") or eng.model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": json_content,
                    },
                    "text": json_content,
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": metrics["prompt_tokens"],
                "completion_tokens": eff_tokens,
                "total_tokens": metrics["prompt_tokens"] + eff_tokens,
            },
            "jev_result": assembled,
            "jev_metrics": metrics,
        }
        return self._json(200, resp, t0=t0, is_stream=False)


def main():
    global ENGINE, STATS_TRACKER
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--worker-socket", default=None, help="use a long-lived model worker via Unix socket")
    ap.add_argument("--devices", default="2,3,0,1")
    ap.add_argument("--budgets", default="")
    ap.add_argument("--max-seq-len", type=int, default=8192)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-graphs", action="store_true")
    ap.add_argument("--offload-experts", nargs="?", const="cpu", default=False, choices=["gpu", "cpu"], help="single-GPU mode: experts in host RAM; 'cpu' computes them on the CPU (default), 'gpu' streams them over PCIe")
    ap.add_argument("--hot-experts", type=int, default=0, help="cpu offload mode: experts per layer kept on the GPU (by usage stats)")
    ap.add_argument("--hot-stats", default="", help="route stats .pt used to pick the hot experts (default: results/route_stats.pt)")
    ap.add_argument("--ep", action="store_true", help="expert parallelism: experts sharded over the devices (e.g. --devices 2,3,0,1 --ep-shards 100,100,100,84)")
    ap.add_argument("--ep-shards", default="", help="experts per device for --ep (default: even split)")
    ap.add_argument(
        "--mtp",
        type=int,
        default=0,
        help="speculative decoding: DSpark drafts verified per step (3-5; 0 = off)",
    )
    ap.add_argument(
        "--mtp-device",
        type=int,
        default=None,
        help="CUDA device used exclusively for DSpark/MTP weights",
    )
    ap.add_argument(
        "--max-seqs",
        type=int,
        default=1,
        help="concurrent sequence slots (1 = serialized single request; >1 = batched decode)",
    )
    a = ap.parse_args()
    try:
        dev_list = [int(d) for d in a.devices.split(",")]
        if any(d < 0 for d in dev_list) or len(set(dev_list)) != len(dev_list):
            raise ValueError
    except ValueError:
        ap.error("--devices must contain distinct non-negative integer indices")
    kw = dict(devices=dev_list, max_seq_len=a.max_seq_len, budgets=parse_budgets(a.budgets),
              use_graphs=not a.no_graphs, offload_experts=a.offload_experts, hot_experts=a.hot_experts, route_stats=a.hot_stats,
              ep=a.ep, ep_shards=[int(v) for v in a.ep_shards.split(",")] if a.ep_shards else None, mtp=a.mtp, mtp_device=a.mtp_device,
              max_seqs=a.max_seqs)
    if a.worker_socket:
        from .gateway_engine import GatewayEngine
        ENGINE = GatewayEngine(a.ckpt, a.worker_socket)
    else:
        ENGINE = Engine(a.ckpt, **kw) if a.ckpt else Engine(**kw)
        try:
            ENGINE.jev_engine.prefix_tree.init_system_prompt()
        except Exception as e:
            print(f"[jev-init] Note: Jev system prompt prefill deferred ({e})", flush=True)
    ENGINE.devices = dev_list
    STATS_TRACKER = StatsTracker(active_devices=dev_list + ([4] if a.worker_socket else []))
    ENGINE.stats_tracker = STATS_TRACKER
    if a.worker_socket:
        STATS_TRACKER.remote_engine = ENGINE
    ThreadingHTTPServer.request_queue_size = 128
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"serving OpenAI-compatible API on http://{a.host}:{a.port}/v1 (model '{ENGINE.model_name}')", flush=True)
    print(f"monitoring dashboard active at http://{a.host}:{a.port}/dashboard", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"[server-fatal] serve_forever error: {e}", flush=True)
        traceback.print_exc()



if __name__ == "__main__":
    main()
