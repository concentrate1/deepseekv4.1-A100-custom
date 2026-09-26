"""Bounded, immutable GPU copies of committed prefix checkpoints.

Never copy live decode slots: MTP may have written rejected future rows into
their rings. A checkpoint includes the compressor, Engram and draft state at
one exact prefix boundary. The Engine lock serializes promotion and restore.
"""
import os
import torch


def enabled(engine):
    return (getattr(engine, "max_seqs", 1) > 1 and bool(getattr(engine, "mtp", 0))
            and os.environ.get("DSV41_GPU_PREFIX_CACHE", "1") != "0"
            and float(os.environ.get("DSV41_GPU_PREFIX_CACHE_GB", "4")) > 0)


def drop(entry):
    entry.pop("gpu_snapshot", None)
    entry.pop("gpu_bytes", None)


def _target_device(kind, holder, key):
    if kind == "scalar_attr":
        return torch.device("cpu")
    return (getattr(holder, key) if kind == "attr" else holder[key]).device


def _available(device):
    # Cached allocator blocks are also available to this process.
    free, _ = torch.cuda.mem_get_info(device)
    return free + max(0, torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device))


def trim(entries, *, needed=None, extra_bytes=0, adding=False):
    """LRU eviction for quota and per-device working-space reserve."""
    limit = max(0, int(float(os.environ.get("DSV41_GPU_PREFIX_CACHE_GB", "4")) * 2**30))
    reserve = max(0, int(float(os.environ.get("DSV41_GPU_PREFIX_RESERVE_GB", "6")) * 2**30))
    max_entries = max(0, int(os.environ.get("DSV41_GPU_PREFIX_CACHE_ENTRIES", "4")))
    resident = sorted((e for e in entries if e.get("gpu_snapshot")),
                      key=lambda e: e.get("last_used", 0))
    devices = {t.device for e in resident for _, _, _, t in e["gpu_snapshot"] if t.is_cuda}
    devices.update(needed or {})
    def fits():
        return (sum(e["gpu_bytes"] for e in resident) + extra_bytes <= limit
                and len(resident) + int(adding) <= max_entries
                and all(_available(d) >= reserve + (needed or {}).get(d, 0) for d in devices))
    while resident and not fits():
        drop(resident.pop(0))
    return fits()


@torch.inference_mode()
def promote(entries, entry):
    """Best-effort promotion from a safe CPU checkpoint, never live state."""
    if entry.get("gpu_snapshot") or not any(e is entry for e in entries):
        return
    needed = {}
    plan = []
    for kind, holder, key, source in entry["snapshot"]:
        device = _target_device(kind, holder, key)
        plan.append((kind, holder, key, source, device))
        if device.type == "cuda":
            needed[device] = needed.get(device, 0) + source.numel() * source.element_size()
    size = sum(needed.values())
    if not size or not trim(entries, needed=needed, extra_bytes=size, adding=True):
        return
    snapshot = []
    try:
        for kind, holder, key, source, device in plan:
            # copy=True is essential: no alias to a mutable source or CPU entry.
            snapshot.append((kind, holder, key, source.to(device=device, copy=True)))
        for device in needed:
            torch.cuda.synchronize(device)
    except (RuntimeError, MemoryError) as exc:
        snapshot.clear()
        print(f"[gpu-prefix-cache] promotion skipped: {type(exc).__name__}", flush=True)
        return
    entry["gpu_snapshot"] = snapshot
    entry["gpu_bytes"] = size
    print(f"[gpu-prefix-cache] stored base={len(entry['base_ids'])} bytes={size}", flush=True)
