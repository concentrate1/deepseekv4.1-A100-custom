"""DSpark reload must preserve the main model and reject busy or failed swaps."""
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from .engine import Engine
from .model_worker import _serve_connection


class FakeDSpark:
    targets = [37, 38, 39]

    def __init__(self, *args, draft_temperature):
        self.draft_temperature = draft_temperature
        self.captured = False

    def capture(self, slots):
        assert slots == 1
        self.captured = True


class FakeConnection:
    def __init__(self, request):
        self.request = request
        self.sent = []

    def recv(self):
        return self.request

    def send(self, value):
        self.sent.append(value)

    def close(self):
        pass


class DSparkReloadTests(unittest.TestCase):
    def engine(self, config_path):
        engine = object.__new__(Engine)
        engine.dspark_config_path = str(config_path)
        engine.dspark_config = types.SimpleNamespace(draft_temperature=0.0)
        engine.dspark_version = 1
        engine._dspark_module_name = None
        engine._mtp_configured = engine.mtp = 5
        engine.mtp_device = None
        engine.model = types.SimpleNamespace(
            blocks=[types.SimpleNamespace(device=torch.device("cpu"))],
            args=object(), embed=object(), head=object(), shared=object(),
            collect_main_hidden=(37, 38, 39),
        )
        engine.rt = object()
        engine.ds = FakeDSpark(draft_temperature=0.0)
        engine.ckpt = "fake-checkpoint"
        engine.lock = threading.Lock()
        engine._slot_lock = threading.Lock()
        engine.mtp_stats = {"enabled": True, "status": "completed"}
        engine.current_phase = "idle"
        return engine

    def test_reload_swaps_only_dspark_after_capture(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "dspark.json"
            config.write_text('{"draft_temperature": 0.7}')
            engine = self.engine(config)
            old_model, old_runtime, old_ds = engine.model, engine.rt, engine.ds
            old_entry = {"gpu_snapshot": [object()], "gpu_bytes": 1}
            engine._rolling_prefix_entries = [old_entry]
            engine._slot_tokens = {0: [1, 2, 3]}
            with patch("dsv41.dspark_lifecycle.load_dspark_rows_class", return_value=(FakeDSpark, None)), \
                 patch("dsv41.load.Checkpoint", return_value=object()):
                status = engine.reload_dspark()
            self.assertIs(engine.model, old_model)
            self.assertIs(engine.rt, old_runtime)
            self.assertIsNot(engine.ds, old_ds)
            self.assertTrue(engine.ds.captured)
            self.assertEqual(status["draft_temperature"], 0.7)
            self.assertEqual(status["version"], 2)
            self.assertEqual(engine.model.collect_main_hidden, (37, 38, 39))
            self.assertEqual(engine._rolling_prefix_entries, [])
            self.assertEqual(engine._slot_tokens, {})
            self.assertNotIn("gpu_snapshot", old_entry)

    def test_invalid_config_keeps_old_component(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "dspark.json"
            config.write_text('{"draft_temperature": -1}')
            engine = self.engine(config)
            old_ds = engine.ds
            with self.assertRaises(ValueError):
                engine.reload_dspark()
            self.assertIs(engine.ds, old_ds)
            self.assertEqual(engine.dspark_version, 1)

    def test_failed_capture_keeps_old_component(self):
        class BrokenDSpark(FakeDSpark):
            def capture(self, slots):
                raise RuntimeError("capture failed")

        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "dspark.json"
            config.write_text('{"draft_temperature": 0.0}')
            engine = self.engine(config)
            old_ds = engine.ds
            with patch("dsv41.dspark_lifecycle.load_dspark_rows_class", return_value=(BrokenDSpark, None)), \
                 patch("dsv41.load.Checkpoint", return_value=object()):
                with self.assertRaisesRegex(RuntimeError, "capture failed"):
                    engine.reload_dspark()
            self.assertIs(engine.ds, old_ds)
            self.assertEqual(engine.dspark_version, 1)
            self.assertEqual(engine.current_phase, "idle")

    def test_python_source_can_load_under_isolated_module_name(self):
        from .dspark_lifecycle import load_dspark_rows_class, discard_dspark_module
        cls, module_name = load_dspark_rows_class(reload_code=True)
        try:
            self.assertEqual(cls.__name__, "DSparkRows")
            self.assertTrue(module_name.startswith("dsv41._dspark_live_"))
        finally:
            discard_dspark_module(module_name)

    def test_busy_worker_rejects_reload(self):
        busy = threading.Lock()
        busy.acquire()
        engine = types.SimpleNamespace(reload_dspark=lambda **_: self.fail("called while busy"))
        conn = FakeConnection({"protocol": 1, "operation": "reload_dspark"})
        _serve_connection(conn, engine, busy)
        self.assertEqual(conn.sent[0]["kind"], "error")
        self.assertIn("busy", conn.sent[0]["message"])
        busy.release()


if __name__ == "__main__":
    unittest.main()
