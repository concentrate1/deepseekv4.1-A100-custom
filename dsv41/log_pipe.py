"""Route a long-lived process's stdout into folders named by local date."""
import argparse
from datetime import date
import os
from pathlib import Path
import sys
import time


class DailyLogSink:
    def __init__(self, root: Path, name: str, mirror=None):
        self.root = Path(root)
        self.name = name
        self.mirror = mirror
        self.day = None
        self.file = None

    def write(self, data: bytes, today: date | None = None):
        today = today or date.today()
        day = today.isoformat()
        if day != self.day:
            self.close()
            folder = self.root / day
            folder.mkdir(parents=True, exist_ok=True)
            self.file = (folder / f"{self.name}.log").open("ab", buffering=0)
            self.day = day
            # Replace the symlink atomically so `logs/latest` follows midnight.
            link = self.root / "latest"
            temporary = self.root / f".latest-{os.getpid()}"
            try:
                temporary.unlink(missing_ok=True)
                temporary.symlink_to(day, target_is_directory=True)
                os.replace(temporary, link)
            finally:
                temporary.unlink(missing_ok=True)
        self.file.write(data)
        if self.mirror is not None:
            try:
                self.mirror.write(data)
                self.mirror.flush()
            except BrokenPipeError:
                # Keep writing the log even if a foreground viewer disconnects.
                self.mirror = None

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None


def follow_file(path: Path, sink: DailyLogSink, until_pid: int | None):
    """Bridge an already-running worker that still writes the old flat log."""
    with path.open("rb") as source:
        while True:
            line = source.readline()
            if line:
                sink.write(line)
                continue
            if until_pid is not None:
                try:
                    os.kill(until_pid, 0)
                except ProcessLookupError:
                    break
            time.sleep(0.2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--name", required=True, choices=("gateway", "worker", "maintenance"))
    parser.add_argument("--tee", action="store_true")
    parser.add_argument("--follow", type=Path, help="bridge an existing flat log")
    parser.add_argument("--until-pid", type=int, help="stop following after this process exits")
    args = parser.parse_args()
    if args.until_pid is not None and args.follow is None:
        parser.error("--until-pid requires --follow")
    sink = DailyLogSink(args.root, args.name, sys.stdout.buffer if args.tee else None)
    try:
        if args.follow is not None:
            follow_file(args.follow, sink, args.until_pid)
        else:
            while line := sys.stdin.buffer.readline():
                sink.write(line)
    finally:
        sink.close()


if __name__ == "__main__":
    main()
