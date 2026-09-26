"""Long-lived model process. The public HTTP gateway can restart independently."""
from contextlib import contextmanager
import argparse
import os
import threading
import traceback
import secrets
import stat
from multiprocessing.connection import Listener, Client

from .engine import Engine, GenParams
from .stats import StatsTracker

PROTOCOL = 1


def private_worker_directory(path, *, create=False):
    parent = os.path.dirname(os.path.abspath(path))
    if create:
        os.makedirs(parent, mode=0o700, exist_ok=True)
    info = os.lstat(parent)
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise PermissionError("worker socket requires an owned directory with mode 0700")


def worker_authkey(path, *, create=False):
    """A private directory is the trust boundary for pickle-based local IPC."""
    private_worker_directory(path, create=create)
    key_path = path + ".key"
    if create:
        key = secrets.token_bytes(32)
        temporary = key_path + "." + secrets.token_hex(8)
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            with os.fdopen(fd, "wb") as out:
                out.write(key)
            os.replace(temporary, key_path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return key
    fd = os.open(key_path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as source:
        info = os.fstat(source.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise PermissionError("worker key requires an owned regular file with mode 0600")
        key = source.read(33)
    if len(key) != 32:
        raise ValueError("invalid worker authentication key")
    return key


def private_listener(path):
    previous = os.umask(0o077)
    try:
        return Listener(path, family="AF_UNIX", authkey=worker_authkey(path, create=True))
    finally:
        os.umask(previous)


class GenerationGate:
    """Concurrent batch clients, exclusive model maintenance and legacy calls."""
    def __init__(self):
        self.condition = threading.Condition()
        self.readers = 0
        self.writer = False
        self.waiting_writers = 0

    def acquire(self, blocking=True):
        with self.condition:
            if not blocking and (self.writer or self.readers):
                return False
            self.waiting_writers += 1
            try:
                while self.writer or self.readers:
                    self.condition.wait()
                self.writer = True
                return True
            finally:
                self.waiting_writers -= 1

    def release(self):
        with self.condition:
            self.writer = False
            self.condition.notify_all()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *args):
        self.release()

    @contextmanager
    def batch(self):
        with self.condition:
            while self.writer or self.waiting_writers:
                self.condition.wait()
            self.readers += 1
        try:
            yield
        finally:
            with self.condition:
                self.readers -= 1
                self.condition.notify_all()


def call(path, request, *, stream=False, cancel_event=None):
    conn = Client(path, family="AF_UNIX", authkey=worker_authkey(path))
    completed = False
    monitor_stop = threading.Event()
    cancellation_sent = threading.Event()
    cancel_lock = threading.Lock()
    def send_cancel():
        with cancel_lock:
            if cancellation_sent.is_set():
                return
            cancellation_sent.set()
            try:
                conn.send({"operation": "cancel"})
            except (OSError, EOFError):
                pass
    def watch_cancellation():
        while not monitor_stop.wait(.1):
            if cancel_event.is_set():
                send_cancel()
                return
    monitor = None
    try:
        conn.send({"protocol": PROTOCOL, **request})
        if cancel_event is not None:
            monitor = threading.Thread(target=watch_cancellation, daemon=True)
            monitor.start()
        if stream:
            while True:
                item = conn.recv()
                if item["kind"] == "error":
                    raise RuntimeError(item["message"])
                if item["kind"] == "done":
                    completed = True
                yield item
                if item["kind"] == "done":
                    break
        else:
            item = conn.recv()
            if item["kind"] == "error":
                raise RuntimeError(item["message"])
            yield item
    finally:
        if stream and not completed:
            send_cancel()
        monitor_stop.set()
        if monitor is not None:
            monitor.join(timeout=1)
        conn.close()


def request(path, operation, *, cancel_event=None, **payload):
    iterator = call(path, {"operation": operation, **payload}, cancel_event=cancel_event)
    try:
        return next(iterator)
    finally:
        iterator.close()


def _route_summary(rt):
    """Distinct routed experts in the last B-row verifier step, without exporting IDs."""
    if not getattr(rt, "route_log", False):
        return "route logging disabled"
    eid, _ = rt.route_snapshot()
    if eid.ndim == 2:
        eid = eid.unsqueeze(1)
    shards = [(str(d), start, start + count) for d, (start, count) in rt.shard.items()]
    layers = []
    for layer in eid:
        ids = layer.reshape(-1)
        ids = ids[ids >= 0]
        unique = ids.unique()
        layers.append({"unique": int(unique.numel()),
                       "by_shard": {name: int(((unique >= lo) & (unique < hi)).sum())
                                    for name, lo, hi in shards}})
    counts = [entry["unique"] for entry in layers]
    return {"rows": int(eid.shape[1]), "topk": int(eid.shape[2]),
            "mean_unique": sum(counts) / len(counts),
            "min_unique": min(counts), "max_unique": max(counts),
            "layers": layers}


def _serve_connection(conn, engine, generation_lock):
    try:
        req = conn.recv()
        if req.get("protocol") != PROTOCOL:
            raise ValueError("worker protocol version mismatch")
        op = req.get("operation")
        if op == "info":
            conn.send({"kind": "result", "model_name": engine.model_name,
                       "max_seq_len": engine.max_seq_len, "protocol": PROTOCOL,
                       "max_seqs": engine.max_seqs,
                       "active_generations": getattr(generation_lock, "readers", 0) + int(getattr(generation_lock, "writer", False)),
                       "active_decode_slots": len(getattr(engine, "_active_slots", {})),
                       "mtp_steps": engine.mtp_stats.get("steps", 0),
                       "last_batch_verify_rows": getattr(engine, "last_batch_verify_rows", None),
                       "prefill_interleave": dict(getattr(engine, "prefill_interleave_stats", {})),
                       "current_phase": engine.current_phase,
                       "prefill_route": {k:v for k,v in (getattr(engine, "_prefill_route", None) or {}).items()
                                         if k in ("total_tokens", "reused_tokens", "prefill_type")}
                                         if engine.current_phase == "prefill" else None,
                       "prefill_work": engine._live_prefill_work() if engine.current_phase == "prefill" else None,
                       "pid": os.getpid(), "ckpt": engine.ckpt,
                       "devices": engine.devices, "mtp": engine.mtp,
                       "expert_devices": [sh["device"].index for sh in (engine.model.blocks[0].ffn.ep or [])],
                       "expert_shards": [sh["n"] for sh in (engine.model.blocks[0].ffn.ep or [])],
                       "ep_relay": getattr(engine.rt, "relay", None),
                       "dspark": engine.dspark_status(),
                       "phase_profile_enabled": engine.phase_profile_enabled})
        elif op == "ep_trace":
            if not generation_lock.acquire(blocking=False):
                conn.send({"kind": "error", "message": "worker is busy; retry EP trace after requests finish"})
                return
            try:
                from .ep import trace_report
                value = trace_report(engine.rt)
            finally:
                generation_lock.release()
            conn.send({"kind": "result", "value": value, "pid": os.getpid()})
        elif op == "route_summary":
            if not generation_lock.acquire(blocking=False):
                conn.send({"kind": "error", "message": "worker is busy; retry route summary after requests finish"})
                return
            try:
                value = _route_summary(engine.rt)
            finally:
                generation_lock.release()
            conn.send({"kind": "result", "value": value, "pid": os.getpid()})
        elif op == "set_phase_profile":
            enabled = req.get("enabled")
            if type(enabled) is not bool:
                raise ValueError("enabled must be boolean")
            if not generation_lock.acquire(blocking=False):
                conn.send({"kind": "error", "message": "worker is busy; retry after requests finish"})
                return
            try:
                engine.phase_profile_enabled = enabled
                engine.last_phase_profile = None
            finally:
                generation_lock.release()
            conn.send({"kind": "result", "enabled": enabled, "pid": os.getpid()})
        elif op == "reload_dspark":
            if type(req.get("reload_code", True)) is not bool:
                raise ValueError("reload_code must be boolean")
            # Never interrupt an in-flight generation. New requests can wait
            # while the component's weights and CUDA graph are rebuilt.
            if not generation_lock.acquire(blocking=False):
                conn.send({"kind": "error", "message": "worker is busy; retry DSpark reload after requests finish"})
                return
            try:
                value = engine.reload_dspark(reload_code=req.get("reload_code", True))
            finally:
                generation_lock.release()
            conn.send({"kind": "result", "value": value, "pid": os.getpid()})
        elif op == "cache_stats":
            conn.send({"kind": "result", "value": engine.get_cache_stats()})
        elif op == "jev":
            with generation_lock:
                value = engine.jev_inference(req["prompt"], req["schema"], req["max_batch"])
            conn.send({"kind": "result", "value": value})
        elif op in ("generate", "generate_text", "stream_text"):
            params = GenParams(**req["params"])
            images = req.get("images")
            token_types = req.get("token_types")
            if token_types is not None:
                token_types = token_types.to(engine.model.blocks[0].device)
            batched = getattr(engine, "max_seqs", 1) > 1
            if batched and op == "generate":
                raise ValueError("batch worker supports stream_text and generate_text, not raw generate")
            cancel_event = threading.Event()
            monitor_stop = threading.Event()
            def monitor_disconnect():
                try:
                    while not monitor_stop.is_set():
                        if conn.poll(0.1):
                            message = conn.recv()
                            if message.get("operation") == "cancel":
                                cancel_event.set()
                                return
                except (OSError, EOFError):
                    cancel_event.set()
            monitor = None
            if batched and hasattr(conn, "poll"):
                monitor = threading.Thread(target=monitor_disconnect, daemon=True)
                monitor.start()
            cancel_kwargs = {"cancel_event": cancel_event} if monitor is not None else {}
            guard = generation_lock.batch() if batched else generation_lock
            try:
                with guard:
                    engine._disable_prefix_cache_once = req.get("disable_prefix_cache", False)
                    if op == "generate_text":
                        value = engine.generate_text(req["prompt_ids"], params, images=images,
                                                     token_types=token_types, **cancel_kwargs)
                        conn.send({"kind": "result", "value": value,
                                   "finish_reason": engine.last_finish_reason,
                                   "decode_tok_s": engine.last_decode_tok_s})
                    else:
                        if op == "generate":
                            iterator = engine.generate(req["prompt_ids"], params, images=images, token_types=token_types)
                        else:
                            iterator = engine.stream_text(req["prompt_ids"], params, images=images,
                                                          token_types=token_types, **cancel_kwargs)
                        try:
                            for value in iterator:
                                conn.send({"kind": "item", "value": value})
                        finally:
                            cancel_event.set()
                            iterator.close()
                        conn.send({"kind": "done", "finish_reason": engine.last_finish_reason,
                                   "decode_tok_s": engine.last_decode_tok_s})
            finally:
                monitor_stop.set()
                if monitor is not None:
                    monitor.join(timeout=1)

        else:
            raise ValueError(f"unknown worker operation: {op}")
    except (BrokenPipeError, EOFError, ConnectionResetError):
        pass
    except Exception as exc:
        traceback.print_exc()
        try:
            conn.send({"kind": "error", "message": str(exc)})
        except (BrokenPipeError, EOFError, ConnectionResetError):
            pass
    finally:
        conn.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--socket", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--devices", default="2,0,1,3")
    ap.add_argument("--ep", action="store_true")
    ap.add_argument("--ep-shards", default="")
    ap.add_argument("--ep-devices", default="")
    ap.add_argument("--max-seq-len", type=int, default=1048576)
    ap.add_argument("--max-seqs", type=int, default=1)
    ap.add_argument("--mtp", type=int, default=5)
    ap.add_argument("--mtp-device", type=int, default=4)
    args = ap.parse_args()
    os.umask(0o077)
    path = os.path.abspath(args.socket)
    private_worker_directory(path, create=True)
    if os.path.exists(path):
        try:
            request(path, "info")
        except (OSError, EOFError, ConnectionError):
            os.unlink(path)
        else:
            raise SystemExit(f"model worker already listening at {path}")
    devices = [int(v) for v in args.devices.split(",")]
    engine = Engine(args.ckpt, devices=devices, max_seq_len=args.max_seq_len,
                    max_seqs=args.max_seqs, ep=args.ep,
                    ep_devices=[int(v) for v in args.ep_devices.split(",")] if args.ep_devices else None,
                    ep_shards=[int(v) for v in args.ep_shards.split(",")] if args.ep_shards else None,
                    mtp=args.mtp, mtp_device=args.mtp_device)
    engine.devices = devices
    engine.stats_tracker = StatsTracker(active_devices=devices + [args.mtp_device])
    try:
        engine.jev_engine.prefix_tree.init_system_prompt()
    except Exception as exc:
        print(f"[jev-init] system prompt prefill deferred: {exc}", flush=True)
    listener = private_listener(path)
    os.chmod(path, 0o600)
    print(f"model worker ready at {path} (pid={os.getpid()})", flush=True)
    generation_lock = GenerationGate()
    try:
        while True:
            conn = listener.accept()
            threading.Thread(target=_serve_connection, args=(conn, engine, generation_lock), daemon=True).start()
    finally:
        listener.close()
        if os.path.exists(path):
            os.unlink(path)


if __name__ == "__main__":
    main()
