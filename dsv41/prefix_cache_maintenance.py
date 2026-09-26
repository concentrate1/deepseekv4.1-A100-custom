"""Expire image-prefix snapshots and compact oversized legacy KV block files."""
import argparse
import hashlib
import os
from pathlib import Path
import time

import torch


def _tensor_digest(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()[:32]


def compact_block(path: Path) -> bool:
    """Rewrite a view-backed block with exact-size storage, preserving its hash."""
    tensor = torch.load(path, map_location="cpu", weights_only=False)
    if _tensor_digest(tensor) != path.stem:
        raise ValueError(f"block hash mismatch: {path}")
    logical = tensor.numel() * tensor.element_size()
    if path.stat().st_size <= logical + 65536:
        return False
    compact = tensor.clone()
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        torch.save(compact, temporary)
        check = torch.load(temporary, map_location="cpu", weights_only=False)
        if check.dtype != tensor.dtype or check.shape != tensor.shape or _tensor_digest(check) != path.stem:
            raise ValueError(f"compacted block failed verification: {path}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def clean_once(root: Path, *, now: float | None = None, max_age_s: float = 86400,
               block_grace_s: float = 3600, compact: bool = True) -> dict:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    now = time.time() if now is None else now
    removed_manifests = 0
    removed_blocks = 0
    compacted_blocks = 0
    references = set()
    readable = True

    for path in root.glob("prefix-*.pt"):
        try:
            if now - path.stat().st_mtime > max_age_s:
                path.unlink()
                removed_manifests += 1
                continue
            payload = torch.load(path, map_location="cpu", weights_only=False)
            for spec in payload.get("block_refs", []) or []:
                if spec:
                    references.update(spec.get("refs", []))
        except Exception as exc:
            print(f"[image-cache-cleanup] manifest unreadable path={path} error={exc}", flush=True)
            readable = False

    if readable:
        for path in (root / "blocks").glob("*.pt"):
            try:
                if path.stem not in references:
                    if now - path.stat().st_mtime > block_grace_s:
                        path.unlink()
                        removed_blocks += 1
                    continue
                if compact and path.stat().st_size > 16 * 2**20:
                    compacted_blocks += int(compact_block(path))
            except Exception as exc:
                print(f"[image-cache-cleanup] block skipped path={path} error={exc}", flush=True)

    result = {"manifests_removed": removed_manifests,
              "blocks_removed": removed_blocks, "blocks_compacted": compacted_blocks}
    print(f"[image-cache-cleanup] {result}", flush=True)
    return result


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--max-age-hours", type=float, default=24)
    parser.add_argument("--interval-seconds", type=float, default=3600)
    parser.add_argument("--until-pid", type=int)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    while True:
        clean_once(args.root, max_age_s=args.max_age_hours * 3600)
        if args.once:
            return
        deadline = time.monotonic() + args.interval_seconds
        while time.monotonic() < deadline:
            if args.until_pid is not None and not _process_alive(args.until_pid):
                return
            time.sleep(min(5, max(0, deadline - time.monotonic())))


if __name__ == "__main__":
    main()
