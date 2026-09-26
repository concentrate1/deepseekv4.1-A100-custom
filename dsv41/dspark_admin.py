"""Local DSpark component administration over the worker's private Unix socket."""
import argparse
import json
from pathlib import Path

from .model_worker import request


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("status", "reload"))
    ap.add_argument("--socket", default=str(Path(__file__).resolve().parent.parent / "logs" / "worker" / "mtp-worker.sock"))
    ap.add_argument("--keep-code", action="store_true",
                    help="reread config and weights, but keep the already imported dspark.py code")
    args = ap.parse_args()
    if args.action == "status":
        result = request(args.socket, "info")
        output = {"worker_pid": result["pid"], "dspark": result["dspark"]}
    else:
        result = request(args.socket, "reload_dspark", reload_code=not args.keep_code)
        output = {"worker_pid": result["pid"], "dspark": result["value"]}
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
