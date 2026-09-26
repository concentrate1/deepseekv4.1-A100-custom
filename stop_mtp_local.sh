#!/usr/bin/env bash
# Stop every process started for this repository's local MTP service.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

/usr/bin/python3 - "$@" <<'PY'
from pathlib import Path
import json
import os
import signal
import sys
import time
import urllib.request

if len(sys.argv) > 2 or (len(sys.argv) == 2 and sys.argv[1] not in ("--force", "--status", "--help")):
    raise SystemExit("usage: ./stop_mtp_local.sh [--force|--status|--help]")
mode = sys.argv[1] if len(sys.argv) == 2 else "stop"
if mode == "--help":
    print("usage: ./stop_mtp_local.sh [--force|--status]")
    print("Stops this repository's gateway, worker, cleaner and log pipes.")
    print("Refuses to interrupt active requests unless --force is given.")
    raise SystemExit(0)

root = Path.cwd()
logs = root / "logs"
socket_paths = {str(logs / "worker" / "mtp-worker.sock"), str(logs / "mtp-worker.sock")}
image_cache = str(root / "cache" / "image-prefix")


def process_args(pid: int) -> list[str]:
    try:
        return [part.decode(errors="replace") for part in
                Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0") if part]
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return []


def has_option(args: list[str], flag: str, value: str) -> bool:
    return any(args[i:i + 2] == [flag, value] for i in range(len(args) - 1))


def service_kind(args: list[str]) -> str | None:
    for i in range(len(args) - 1):
        if args[i] != "-m":
            continue
        module = args[i + 1]
        if module == "dsv41.serve" and any(has_option(args, "--worker-socket", path) for path in socket_paths):
            return "gateway"
        if module == "dsv41.model_worker" and any(has_option(args, "--socket", path) for path in socket_paths):
            return "worker"
        if module == "dsv41.prefix_cache_maintenance" and has_option(args, "--root", image_cache):
            return "maintenance"
        if module == "dsv41.log_pipe" and has_option(args, "--root", str(logs)):
            return "log_pipe"
    return None


def discover() -> dict[int, str]:
    found = {}
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            pid = int(entry.name)
            kind = service_kind(process_args(pid))
            if kind:
                found[pid] = kind
    return found


def running(pid: int) -> bool:
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except (FileNotFoundError, ProcessLookupError):
        return False


def active_requests(found: dict[int, str]) -> tuple[int | None, str]:
    gateways = [pid for pid, kind in found.items() if kind == "gateway"]
    if not gateways:
        return 0, "gateway not running"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    count = 0
    for pid in gateways:
        args = process_args(pid)
        host = next((args[i + 1] for i in range(len(args) - 1) if args[i] == "--host"), "host.docker.internal")
        port = next((args[i + 1] for i in range(len(args) - 1) if args[i] == "--port"), "40033")
        if host in ("0.0.0.0", "::"):
            host = "127.0.0.1"
        try:
            with opener.open(f"http://{host}:{port}/api/metrics", timeout=3) as response:
                metrics = json.load(response)
        except Exception as exc:
            return None, f"cannot inspect gateway requests: {exc}"
        count += max(0, int(metrics.get("active_clients", 0)))
        if metrics.get("engine_phase") in ("prefill", "decode") or any(
            slot.get("status") in ("prefilling", "generating")
            for slot in metrics.get("slots", [])
        ):
            count = max(1, count)
    return count, f"active requests: {count}"


found = discover()
count, status = active_requests(found)
if mode == "--status":
    for pid, kind in sorted(found.items()):
        print(f"{kind}: PID {pid}")
    print(status)
    raise SystemExit(0 if count is not None else 1)
if mode != "--force" and (count is None or count > 0):
    raise SystemExit(f"refusing to stop: {status}; retry after requests finish or use --force")
for kind in ("gateway", "worker", "maintenance", "log_pipe"):
    group = [pid for pid, actual in found.items() if actual == kind]
    for pid in group:
        if running(pid):
            try:
                os.kill(pid, signal.SIGTERM)
                print(f"stopping {kind}: PID {pid}", flush=True)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 10
    while any(running(pid) for pid in group) and time.monotonic() < deadline:
        time.sleep(0.1)
    for pid in group:
        if running(pid):
            try:
                os.kill(pid, signal.SIGKILL)
                print(f"force-stopped {kind}: PID {pid}", flush=True)
            except ProcessLookupError:
                pass

remaining = discover()
if remaining:
    raise SystemExit(f"service processes remain: {remaining}")
for name in ("server.pid", "worker.pid", "image-cache-cleaner.pid", "worker/mtp-worker.sock", "worker/mtp-worker.sock.key", "mtp-worker.sock"):
    (logs / name).unlink(missing_ok=True)
print("local MTP service stopped")
PY
