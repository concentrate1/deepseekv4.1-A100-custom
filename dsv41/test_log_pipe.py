"""Long-running logs must rotate into a new folder at midnight."""
from datetime import date
from pathlib import Path
import tempfile
import unittest

from .log_pipe import DailyLogSink


class DailyLogSinkTest(unittest.TestCase):
    def test_daily_folders_and_latest_link(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sink = DailyLogSink(root, "gateway")
            try:
                sink.write(b"first day\n", date(2026, 9, 23))
                sink.write(b"next day\n", date(2026, 9, 24))
            finally:
                sink.close()
            self.assertEqual((root / "2026-09-23/gateway.log").read_bytes(), b"first day\n")
            self.assertEqual((root / "2026-09-24/gateway.log").read_bytes(), b"next day\n")
            self.assertEqual((root / "latest").resolve(), root / "2026-09-24")


if __name__ == "__main__":
    unittest.main()
