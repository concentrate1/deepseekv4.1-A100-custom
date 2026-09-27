#!/usr/bin/env bash
# Single-stream DSpark/MTP profile for this 5x-A100 NUMA island.
# Backbone pipeline: cuda:2 -> cuda:0 -> cuda:1 -> cuda:3
# DSpark + vision tower: cuda:4 (cuda:3 <-> cuda:4 is a PIX link)
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

case "${1:---foreground}" in
    --foreground) START_MODE=foreground ;;
    --background) START_MODE=background ;;
    --help)
        echo "Usage: $0 [--background|--foreground]"
        echo "Logs: logs/latest/gateway.log and logs/latest/worker.log"
        exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
esac
if (( $# > 1 )); then
    echo "Expected at most one option." >&2
    exit 2
fi

# Startup configuration. Keep profile selection and all environment settings
# together before logging, port checks, or process creation.
# The batch wrapper shares lifecycle/PID files so profiles cannot coexist accidentally.
DSV41_LOCAL_PROFILE="${DSV41_LOCAL_PROFILE:-single}"
case "$DSV41_LOCAL_PROFILE" in
    single) LOCAL_MAX_SEQS=1; LOCAL_INIT_TOKENS=65536 ;;
    batch) LOCAL_MAX_SEQS=3; LOCAL_INIT_TOKENS=32768 ;;
    *) echo "Unknown local profile: $DSV41_LOCAL_PROFILE" >&2; exit 2 ;;
esac
export DSV41_LOCAL_PROFILE
DSV41_MTP_PYTHON="${DSV41_PYTHON:-/usr/bin/python3}"
DSV41_EP_DEVICES="${DSV41_EP_DEVICES:-2,0,1,3}"
DSV41_EP_SHARDS="${DSV41_EP_SHARDS:-92,99,98,95}"
# CUDA indices are used directly by this profile; hide unrelated GPUs 5-9.
export CUDA_VISIBLE_DEVICES=0,1,2,3,4
DSV41_MTP_HOST="${DSV41_HOST:-host.docker.internal}"
DSV41_MTP_PORT="${DSV41_PORT:-40033}"
DSV41_MTP_DRAFTS="${DSV41_MTP_DRAFTS:-4}"
case "$DSV41_MTP_DRAFTS" in
    0|3|4|5) ;;
    *) echo "DSV41_MTP_DRAFTS must be 0, 3, 4, or 5" >&2; exit 2 ;;
esac
DSV41_EXPERT_PARALLEL="${DSV41_EXPERT_PARALLEL:-1}"
case "$DSV41_EXPERT_PARALLEL" in
    0|1) ;;
    *) echo "DSV41_EXPERT_PARALLEL must be 0 or 1" >&2; exit 2 ;;
esac
ep_args=()
if [[ "$DSV41_EXPERT_PARALLEL" == 1 ]]; then
    ep_args=(--ep --ep-shards "$DSV41_EP_SHARDS" --ep-devices "$DSV41_EP_DEVICES")
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DSV41_VISION_DEVICE=cuda:4
export DSV41_DSPARK_CONFIG="${DSV41_DSPARK_CONFIG:-${PWD}/config/dspark_runtime.json}"

# Four contiguous 10-layer pipeline stages. The physical expert allocation
# preserves the current machine's memory balance while following device order.
export DSV41_LAYER_COUNTS=10,10,10,10
export DSV41_MAX_SEQS="$LOCAL_MAX_SEQS"
# With one sequence, every B5 verifier row reads the same index cache.
# Share it directly to avoid five copies of the preallocated 107K/215K rows.
export DSV41_INDEXER_SHARED="${DSV41_INDEXER_SHARED:-1}"
if [[ "$DSV41_LOCAL_PROFILE" == batch ]]; then
    export DSV41_INDEXER_SHARED=0
fi
case "$DSV41_INDEXER_SHARED" in
    0|1) ;;
    *) echo "DSV41_INDEXER_SHARED must be 0 or 1" >&2; exit 2 ;;
esac
export DSV41_MOE_PREFILL_CHUNK=2048
export DSV41_ENGRAM_PREFILL_CHUNK=512
export DSV41_HC_PREFILL_CHUNK=256
export DSV41_SPARSE_ATTN_CHUNK=128
export DSV41_EP_GRAPH_TOKENS=1048576
export DSV41_EP_CAND_TOKENS=1048576
export DSV41_CACHE_INIT_TOKENS="$LOCAL_INIT_TOKENS"
export DSV41_EP_PREALLOC_TOKENS="$LOCAL_INIT_TOKENS"
export DSV41_EXACT_CACHE_GROW=1
export DSV41_EP_COMPACT_XQ=1
# 默认关闭直接槽位续算，不影响其他前缀缓存；MTP 模式不使用这条路径。
export DSV41_LIVE_SLOT_REUSE="${DSV41_LIVE_SLOT_REUSE:-0}"
export DSV41_PREFIX_DEDUP_MIRRORS=1
export DSV41_CED=1

# Keep DSpark enabled throughout the entire valid 1M context range. A request
# longer than --max-seq-len is rejected before it could reach this boundary.
export DSV41_MTP_LONG_PROMPT_LIMIT=1048576
export DSV41_MTP_UNLOAD_ON_LONG=0

export DSV41_INTERACTIVE_MAX_NEW=65536
export DSV41_LONG_PROMPT_MAX_NEW=65536

export DSV41_PREFIX_CACHE_DIR=/dev/shm/dsv41-prefix-cache
export DSV41_IMAGE_PREFIX_CACHE_DIR="${DSV41_IMAGE_PREFIX_CACHE_DIR:-${PWD}/cache/image-prefix}"
export DSV41_IMAGE_PREFIX_CACHE_ENTRIES="${DSV41_IMAGE_PREFIX_CACHE_ENTRIES:-8}"
export DSV41_IMAGE_PREFIX_CACHE_GB="${DSV41_IMAGE_PREFIX_CACHE_GB:-8}"
export DSV41_PREFIX_CACHE_ENTRIES=16
export DSV41_PREFIX_CACHE_GB=32
export DSV41_PREFIX_TMPFS_ENTRIES=16
export DSV41_PREFIX_TMPFS_GB=32
export DSV41_PREFIX_BLOCK_REPLAY=1
export DSV41_PREFIX_BLOCK_SIZE=512
export DSV41_PREFIX_BLOCK_MIN=16
export DSV41_PREFIX_ANCHOR_STRIDE=1024
# One fresh anchor is enough for batch continuations; retain older cache entries
# without copying and persisting a second large snapshot in the same request.
if [[ "$DSV41_LOCAL_PROFILE" == batch ]]; then
    export DSV41_PREFIX_ANCHOR_MAX="${DSV41_PREFIX_ANCHOR_MAX:-1}"
else
    export DSV41_PREFIX_ANCHOR_MAX="${DSV41_PREFIX_ANCHOR_MAX:-2}"
fi

export DSV41_REPETITION_PENALTY=1.0
export DSV41_FREQUENCY_PENALTY=0.0
export DSV41_PRESENCE_PENALTY=0.0
export DSV41_PENALTY_WINDOW=2048
export DSV41_PROGRESSIVE_PENALTY=0.0
export DSV41_BAN_CYCLES=0
export DSV41_LOOP_DETECT=0
export DSV41_MIN_LOOP_MATCH=48
export DSV41_MIN_LOOP_CYCLE=1

DEFAULT_CKPT="/models/DeepSeek-V4.1-Flash"
CKPT="${DSV41_CKPT:-$DEFAULT_CKPT}"
WORKER_SOCKET="${DSV41_WORKER_SOCKET:-${PWD}/logs/worker/mtp-worker.sock}"

# Runtime checks and process lifecycle.
mkdir -p logs
if [[ "$START_MODE" == foreground ]]; then
    # One pipe groups logs by day and mirrors foreground output to the console.
    exec > >( "$DSV41_MTP_PYTHON" -u -m dsv41.log_pipe --root "$PWD/logs" --name gateway --tee ) 2>&1
fi
# Wait briefly for a gateway that has just received SIGTERM to release its
# listener. A genuinely occupied port still fails before any model is loaded.
port_ready=0
for ((attempt=0; attempt<6; attempt++)); do
    if pgrep -af 'python.*-m dsv41[.]serve' | grep -Fq -- "--port $DSV41_MTP_PORT"; then
        :
    elif "$DSV41_MTP_PYTHON" - "$DSV41_MTP_HOST" "$DSV41_MTP_PORT" <<'PYCHECK'
import socket
import sys
host, port = sys.argv[1], int(sys.argv[2])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    raise SystemExit(0 if sock.connect_ex((host, port)) != 0 else 1)
PYCHECK
    then
        port_ready=1
        break
    fi
    sleep 1
done
if (( port_ready == 0 )); then
    echo "Refusing to start: ${DSV41_MTP_HOST}:${DSV41_MTP_PORT} is already in use." >&2
    exit 1
fi

if [[ ! -d "$CKPT" ]]; then
    echo "Checkpoint directory not found: $CKPT" >&2
    exit 1
fi

if [[ "$START_MODE" == background ]]; then
    # A new session survives the short-lived shell that requested startup.
    # The foreground child writes its actual PID to logs/server.pid.
    setsid -f "$0" --foreground > /dev/null 2>&1 < /dev/null
    echo "Service starting. Logs: logs/latest/gateway.log and logs/latest/worker.log"
    exit 0
fi

printf '%s\n' "$$" > logs/server.pid

# Keep the model worker alive when the HTTP gateway exits or is restarted.
# A worker that is still loading has no socket yet; do not launch a second one.
if ! "$DSV41_MTP_PYTHON" - "$WORKER_SOCKET" <<'PYCHECK' >/dev/null 2>&1
import sys
from dsv41.model_worker import request
request(sys.argv[1], "info")
PYCHECK
then
    worker_loading=0
    if [[ -f logs/worker.pid ]]; then
        worker_pid="$(cat logs/worker.pid)"
        if [[ "$worker_pid" =~ ^[0-9]+$ ]] && ps -p "$worker_pid" -o args= 2>/dev/null | grep -Fq -- "--socket $WORKER_SOCKET"; then
            worker_loading=1
        fi
    fi
    if (( worker_loading == 0 )) && pgrep -af 'python.*-m dsv41[.]model_worker' >/dev/null; then
        echo "Another model worker is running but not at $WORKER_SOCKET; inspect it before starting a second model." >&2
        exit 1
    fi
    if (( worker_loading == 0 )); then
        # A process can exit before its large CUDA context finishes releasing.
        # Wait for the dedicated five-card profile's memory before loading again.
        "$DSV41_MTP_PYTHON" - <<'PYCHECK'
import subprocess
import time
end = time.monotonic() + 60
announced = False
while True:
    output = subprocess.check_output([
        "nvidia-smi", "--id=0,1,2,3,4", "--query-gpu=memory.free",
        "--format=csv,noheader,nounits",
    ], text=True)
    free = [int(value) for value in output.splitlines()]
    if len(free) == 5 and min(free) >= 75000:
        break
    if not announced:
        print(f"Waiting for GPU memory release before model load: free MiB={free}", flush=True)
        announced = True
    if time.monotonic() >= end:
        raise SystemExit(f"GPU memory remains occupied (free MiB={free}); inspect it before retrying.")
    time.sleep(1)
PYCHECK
        nohup "$DSV41_MTP_PYTHON" -u -m dsv41.model_worker \
            --socket "$WORKER_SOCKET" \
            --ckpt "$CKPT" \
            --devices 2,0,1,3 \
            "${ep_args[@]}" \
            --max-seq-len 1048576 --max-seqs "$LOCAL_MAX_SEQS" \
            --mtp "$DSV41_MTP_DRAFTS" --mtp-device 4 \
            > >( "$DSV41_MTP_PYTHON" -u -m dsv41.log_pipe --root "$PWD/logs" --name worker ) 2>&1 < /dev/null &
        printf '%s\n' "$!" > logs/worker.pid
        echo "Model worker started (pid $(cat logs/worker.pid)); waiting for checkpoint load..."
    else
        echo "Model worker is loading; waiting for its socket..."
    fi
    worker_pid="$(cat logs/worker.pid)"
    ready=0
    for ((attempt=0; attempt<180; attempt++)); do
        if "$DSV41_MTP_PYTHON" - "$WORKER_SOCKET" <<'PYCHECK' >/dev/null 2>&1
import sys
from dsv41.model_worker import request
request(sys.argv[1], "info")
PYCHECK
        then
            ready=1
            break
        fi
        worker_state="$(ps -p "$worker_pid" -o stat= 2>/dev/null || true)"
        if [[ -z "$worker_state" || "$worker_state" == Z* ]]; then
            echo "Model worker exited while loading. Check logs/latest/worker.log" >&2
            exit 1
        fi
        sleep 5
    done
    if (( ready == 0 )); then
        echo "Model worker did not become ready. Check logs/latest/worker.log" >&2
        exit 1
    fi
fi

# Reusing an incompatible worker would silently keep the old concurrency/MTP profile.
"$DSV41_MTP_PYTHON" - "$WORKER_SOCKET" "$CKPT" "$LOCAL_MAX_SEQS" "$DSV41_MTP_DRAFTS" "$DSV41_EXPERT_PARALLEL" "$DSV41_EP_DEVICES" "$DSV41_EP_SHARDS" <<'PYCHECK'
import os
import sys
from dsv41.model_worker import request
info = request(sys.argv[1], "info")
expected = (os.path.realpath(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), 1048576, [2, 0, 1, 3])
expected += ([int(x) for x in sys.argv[6].split(",")] if sys.argv[5] == "1" else [],
             [int(x) for x in sys.argv[7].split(",")] if sys.argv[5] == "1" else [])
actual = (os.path.realpath(info["ckpt"]), info.get("max_seqs", 1), info["mtp"], info["max_seq_len"], info["devices"], info.get("expert_devices"), info.get("expert_shards"))
if actual != expected:
    raise SystemExit(f"Worker profile mismatch: actual={actual}, requested={expected}. "
                     "Arrange a full worker restart to switch profiles; no process was stopped.")
PYCHECK

# Keep a small independent janitor alongside the model. It expires image
# snapshots after 24 hours and exits when this worker process exits.
cleaner_running=0
if [[ -f logs/image-cache-cleaner.pid ]]; then
    cleaner_pid="$(cat logs/image-cache-cleaner.pid)"
    if [[ "$cleaner_pid" =~ ^[0-9]+$ ]] && ps -p "$cleaner_pid" -o args= 2>/dev/null | grep -Fq -- '-m dsv41.prefix_cache_maintenance'; then
        cleaner_running=1
    fi
fi
if (( cleaner_running == 0 )); then
    worker_pid="$(cat logs/worker.pid)"
    nohup env CUDA_VISIBLE_DEVICES="" "$DSV41_MTP_PYTHON" -u -m dsv41.prefix_cache_maintenance \
        --root "$DSV41_IMAGE_PREFIX_CACHE_DIR" --max-age-hours 24 \
        --interval-seconds 3600 --until-pid "$worker_pid" \
        > >( "$DSV41_MTP_PYTHON" -u -m dsv41.log_pipe --root "$PWD/logs" --name maintenance ) 2>&1 < /dev/null &
    printf '%s\n' "$!" > logs/image-cache-cleaner.pid
fi

echo "Model worker ready; starting HTTP gateway on ${DSV41_MTP_HOST}:${DSV41_MTP_PORT}"
# The gateway only tokenizes and formats CPU inputs; it needs no CUDA context.
CUDA_VISIBLE_DEVICES="" exec "$DSV41_MTP_PYTHON" -u -m dsv41.serve \
    --worker-socket "$WORKER_SOCKET" \
    --ckpt "$CKPT" \
    --devices 2,0,1,3 \
    --host "$DSV41_MTP_HOST" \
    --port "$DSV41_MTP_PORT"
