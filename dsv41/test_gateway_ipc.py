"""IPC regression: gateway and worker must not share a process auth key."""
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import torch

from . import gateway_engine  # registers inline CPU tensor serialization
from .engine import GenParams
from .model_worker import request
from .vision import ImageInput


_FAKE_WORKER = r'''
import sys
import threading
import torch
from multiprocessing.connection import Listener
from dsv41.model_worker import private_listener, _serve_connection

class FakeEngine:
    model = type("Model", (), {"blocks": [type("Block", (), {"device": torch.device("cpu")})()]})()
    last_finish_reason = "stop"
    last_decode_tok_s = 1.0

    def generate_text(self, ids, params, images=None, token_types=None):
        assert images[0][0].patches.dtype == torch.bfloat16
        assert token_types.dtype == torch.int64
        assert images[0][0].patches.shape == (1, 3, 14, 14)
        return "image received", 1

listener = private_listener(sys.argv[1])
_serve_connection(listener.accept(), FakeEngine(), threading.Lock())
listener.close()
'''


_FAKE_STREAM_WORKER = r'''
import sys, threading, torch
from dsv41.model_worker import private_listener, _serve_connection, GenerationGate
class FakeEngine:
    max_seqs = 3
    model = type("Model", (), {"blocks": [type("Block", (), {"device": torch.device("cpu")})()]})()
    last_finish_reason = "cancelled"
    last_decode_tok_s = 0.0
    cancelled = False
    def stream_text(self, ids, params, cancel_event=None, **kw):
        yield "hello", None
        self.cancelled = cancel_event.wait(3)
        if not self.cancelled:
            raise RuntimeError("cancel not delivered")
        yield "hello", 1
engine = FakeEngine()
listener = private_listener(sys.argv[1])
_serve_connection(listener.accept(), engine, GenerationGate())
listener.close()
assert engine.cancelled
print("cancel acknowledged", flush=True)
'''


class GatewayIPCTest(unittest.TestCase):
    def test_stream_disconnect_notifies_worker_without_waiting_for_generation_limit(self):
        from .model_worker import call
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp)/"stream.sock")
            process = subprocess.Popen([sys.executable, "-c", _FAKE_STREAM_WORKER, path],
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            try:
                for _ in range(100):
                    if Path(path).exists():
                        break
                    if process.poll() is not None:
                        self.fail(process.stdout.read())
                    time.sleep(.05)
                stream = call(path, {"operation": "stream_text", "prompt_ids": [1],
                                    "params": vars(GenParams(max_new_tokens=65536))}, stream=True)
                self.assertEqual(next(stream)["value"], ("hello", None))
                stream.close()
                output, _ = process.communicate(timeout=5)
                self.assertEqual(process.returncode, 0, output)
                self.assertIn("cancel acknowledged", output)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=5)
                process.stdout.close()

    def test_nonstream_cancellation_event_crosses_worker_socket(self):
        import threading
        code = _FAKE_STREAM_WORKER.replace('def stream_text(self, ids, params, cancel_event=None, **kw):',
                                          'def generate_text(self, ids, params, cancel_event=None, **kw):')
        code = code.replace('        yield "hello", None',
                            '        open(sys.argv[1] + ".admitted", "w").close()')
        code = code.replace('        yield "hello", 1', '        return "cancelled", 0')
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp)/'nonstream.sock')
            process = subprocess.Popen([sys.executable, '-c', code, path], stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True)
            cancellation, values, errors = threading.Event(), [], []
            def client():
                try:
                    values.append(request(path, 'generate_text', cancel_event=cancellation,
                                          prompt_ids=[1], params=vars(GenParams(max_new_tokens=65536))))
                except Exception as exc:
                    errors.append(exc)
            thread = None
            try:
                for _ in range(100):
                    if Path(path).exists():break
                    if process.poll() is not None:self.fail(process.stdout.read())
                    time.sleep(.05)
                thread = threading.Thread(target=client)
                thread.start()
                for _ in range(100):
                    if Path(path+'.admitted').exists():break
                    time.sleep(.02)
                self.assertTrue(Path(path+'.admitted').exists())
                cancellation.set()
                thread.join(5)
                self.assertFalse(thread.is_alive())
                self.assertEqual(errors, [])
                self.assertEqual(values[0]['value'], ('cancelled',0))
                output, _ = process.communicate(timeout=5)
                self.assertEqual(process.returncode,0,output)
            finally:
                cancellation.set()
                if process.poll() is None:
                    process.kill();process.wait(timeout=5)
                if thread is not None:thread.join(5)
                process.stdout.close()

    def test_first_token_speed_is_not_reported_as_decode_throughput(self):
        import threading
        engine = gateway_engine.GatewayEngine.__new__(gateway_engine.GatewayEngine)
        engine._state = threading.local()
        engine._remember({"finish_reason": "stop", "decode_tok_s": 7452.3}, 1)
        self.assertEqual(engine.last_decode_tok_s, 0.0)
        engine._remember({"finish_reason": "stop", "decode_tok_s": 7452.3}, 2)
        self.assertIsNone(engine.last_decode_tok_s)


    def test_cpu_image_tensors_cross_independent_processes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "worker.sock")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = ""
            process = subprocess.Popen(
                [sys.executable, "-c", _FAKE_WORKER, path],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env,
            )
            try:
                for _ in range(100):
                    if Path(path).exists():
                        break
                    if process.poll() is not None:
                        self.fail(f"fake worker exited: {process.stdout.read()}")
                    time.sleep(0.05)
                else:
                    self.fail("fake worker socket not created")
                image = ImageInput(1, torch.ones((1, 3, 14, 14), dtype=torch.bfloat16),
                                   1, 1, torch.tensor([1, 2]))
                result = request(path, "generate_text", prompt_ids=[1],
                                 params=vars(GenParams()), images=[[image]],
                                 token_types=torch.tensor([[1, 2]]))
                self.assertEqual(result["value"], ("image received", 1))
                self.assertEqual(process.wait(timeout=5), 0, process.stdout.read())
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=5)
                process.stdout.close()


if __name__ == "__main__":
    unittest.main()
