"""Private worker controls used for brief, opt-in performance diagnosis."""
import argparse
import json
from pathlib import Path

from .model_worker import request


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("profile-on", "profile-off", "ep-trace", "route-stats", "status"))
    ap.add_argument("--socket", default=str(Path(__file__).resolve().parent.parent / "logs" / "worker" / "mtp-worker.sock"))
    args = ap.parse_args()
    if args.action == "status":
        result = request(args.socket, "info")
        output = {"worker_pid": result["pid"],
                  "phase_profile_enabled": result["phase_profile_enabled"]}
    elif args.action == "ep-trace":
        result = request(args.socket, "ep_trace")
        output = {"worker_pid": result["pid"], "trace": result["value"]}
    elif args.action == "route-stats":
        result = request(args.socket, "route_summary")
        output = {"worker_pid": result["pid"], "route_summary": result["value"]}
    else:
        result = request(args.socket, "set_phase_profile", enabled=args.action == "profile-on")
        output = {"worker_pid": result["pid"], "phase_profile_enabled": result["enabled"]}
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
