"""Request, stream and bounded-image regressions without loading model weights."""
import base64
import io
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from http.server import ThreadingHTTPServer
from urllib.request import build_opener, ProxyHandler, Request

from . import serve, vision
from .engine import Engine, GenParams, _BatchRequest
from .model_worker import worker_authkey, private_listener


class ServiceBoundaryTests(unittest.TestCase):
    def test_invalid_requests_return_400_before_engine_dispatch(self):
        invalid = [[], None, {'max_tokens': 0}, {'max_tokens': 'abc'},
                   {'temperature': None}, {'top_p': float('nan')},
                   {'prompt': []}, {'prompt': [1, 2]}, {'response_format': []},
                   {'response_format': {'json_schema': []}}, {'messages': [1]},
                   {'stream_options': []}, {'stop': [1]}, {'schema': 1}]
        for body in invalid:
            with self.subTest(body=body):
                handler = serve.Handler.__new__(serve.Handler)
                raw = json.dumps(body).encode()
                handler.rfile = io.BytesIO(raw)
                handler.headers = {'Content-Length': str(len(raw))}
                handler.path = '/v1/chat/completions'
                replies = []
                handler._json = lambda status, value, **kw: replies.append((status, value))
                with patch.object(serve, 'STATS_TRACKER', None):
                    handler.do_POST()
                self.assertEqual(replies[0][0], 400)

    def test_null_response_format_keeps_normal_chat_dispatch(self):
        handler = serve.Handler.__new__(serve.Handler)
        raw = b'{"response_format":null}'
        handler.rfile = io.BytesIO(raw)
        handler.headers = {'Content-Length': str(len(raw))}
        handler.path = '/v1/chat/completions'
        received = []
        handler._chat = received.append
        with patch.object(serve, 'STATS_TRACKER', None), patch.object(serve, 'ENGINE', SimpleNamespace(model_name='test')):
            handler.do_POST()
        self.assertEqual(received[0]['response_format'], {'json_schema': {}})

    def test_remote_and_base64_inputs_are_bounded(self):
        class Response(io.BytesIO):
            def read(self, size=-1):
                self.assertion(size)
                return super().read(size)
            @staticmethod
            def assertion(size):
                if size <= 0 or size > 65536:
                    raise AssertionError('unbounded read')
        with patch.dict(os.environ, {'DSV41_IMAGE_MAX_BYTES': '5'}):
            for value in (b'12345', b'123456'):
                with patch.object(vision, 'urlopen', return_value=Response(value)):
                    if len(value) == 5:
                        self.assertEqual(vision.load_image_bytes({'url': 'http://example.test/image'}), value)
                    else:
                        with self.assertRaises(ValueError):
                            vision.load_image_bytes({'url': 'http://example.test/image'})
                if len(value) > 5:
                    with self.assertRaises(ValueError):
                        vision.load_image_bytes({'data': base64.b64encode(value).decode()})
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/'image'
                path.write_bytes(b'123456')
                with self.assertRaises(ValueError):
                    vision.load_image_bytes({'url': str(path)})
        with patch.dict(os.environ, {'DSV41_IMAGE_MAX_PIXELS': '10'}):
            with self.assertRaises(ValueError):
                vision.check_image_pixels(SimpleNamespace(width=4, height=3))

    def test_worker_directory_and_key_are_private(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory)/'worker.sock')
            first = worker_authkey(path, create=True)
            self.assertEqual(worker_authkey(path), first)
            self.assertEqual(Path(path+'.key').stat().st_mode & 0o777, 0o600)
            self.assertNotEqual(worker_authkey(path, create=True), first)
            Path(path+'.key').chmod(0o644)
            with self.assertRaises(PermissionError):
                worker_authkey(path)
            Path(directory).chmod(0o755)
            with self.assertRaises(PermissionError):
                worker_authkey(path, create=True)

    def test_socket_is_private_at_creation_even_with_permissive_umask(self):
        from multiprocessing.connection import Listener
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory)/'private'/'worker.sock')
            observed = []
            def create(address, **kw):
                listener = Listener(address, **kw)
                observed.append(Path(address).stat().st_mode & 0o777)
                return listener
            old = os.umask(0)
            try:
                with patch('dsv41.model_worker.Listener', side_effect=create):
                    listener = private_listener(path)
                listener.close()
            finally:
                os.umask(old)
            self.assertEqual(observed, [0o700])
            self.assertEqual(Path(path).parent.stat().st_mode & 0o777, 0o700)

    def test_cancelled_active_and_queued_requests_release_resources(self):
        import queue
        engine = Engine.__new__(Engine)
        engine._slot_lock = threading.Lock()
        engine._slot_tokens_lock = threading.Lock()
        engine._slot_tokens = {1: [1]}
        engine.slot_states = {1: {'status': 'generating'}}
        engine._free_decode_slots = [2]
        engine._batch_queue = queue.Queue()
        active = _BatchRequest([1], GenParams(), 10, None)
        active.slot_id = 1
        active.images, active.main_h = ['image'], object()
        engine._active_slots = {1: active}
        engine._cancel_batch_request(active)
        self.assertTrue(active.done_event.is_set())
        self.assertEqual(engine._active_slots, {})
        self.assertEqual(engine._free_decode_slots, [2, 1])
        self.assertNotIn(1, engine._slot_tokens)
        self.assertIsNone(active.main_h)
        live = _BatchRequest([2], GenParams(), 10, None)
        cancelled = _BatchRequest([3], GenParams(), 10, None)
        cancelled.cancel_event.set()
        engine._batch_queue.put(live)
        engine._batch_queue.put(cancelled)
        engine._discard_cancelled_queued()
        self.assertTrue(cancelled.done_event.is_set())
        self.assertIs(engine._batch_queue.get_nowait(), live)
        self.assertTrue(engine._batch_queue.empty())

    def test_scheduler_acknowledges_cancelled_requests_without_gpu_dispatch(self):
        import queue
        engine = Engine.__new__(Engine)
        engine.rt = SimpleNamespace(B=5)
        engine._slot_lock = threading.Lock()
        engine._slot_tokens = {1: [1]}
        engine._slot_tokens_lock = threading.Lock()
        engine.slot_states = {1: {"status": "generating"}}
        engine.current_phase = "decode"
        engine._free_decode_slots = [2]
        engine._batch_queue = queue.Queue()
        engine._batch_stop_event = threading.Event()
        active = _BatchRequest([1], GenParams(), 65536, None)
        active.slot_id = 1
        active.cancel_event.set()
        queued = _BatchRequest([2], GenParams(), 65536, None)
        queued.cancel_event.set()
        engine._active_slots = {1: active}
        engine._batch_queue.put(queued)
        thread = threading.Thread(target=engine._batch_worker_loop)
        thread.start()
        try:
            self.assertTrue(active.done_event.wait(2))
            self.assertTrue(queued.done_event.wait(2))
            self.assertEqual(engine.current_phase, "idle")
            self.assertEqual(engine._active_slots, {})
        finally:
            engine._batch_stop_event.set()
            thread.join(2)
        self.assertFalse(thread.is_alive())

    def test_completion_closes_generator_on_http_disconnect(self):
        closed = []
        def stream(*args, **kw):
            try:
                yield 'hello', None
                yield 'hello', 1
            finally:
                closed.append(True)
        engine = SimpleNamespace(model_name='test', tok=SimpleNamespace(encode=lambda _: [1]), stream_text=stream)
        handler = serve.Handler.__new__(serve.Handler)
        handler.send_response = lambda *_: None
        handler.send_header = lambda *_: None
        handler.end_headers = lambda: None
        class Disconnected:
            def write(self, data):
                raise BrokenPipeError()
        handler.wfile = Disconnected()
        with patch.object(serve, 'ENGINE', engine), patch.object(serve, 'STATS_TRACKER', None):
            handler._completion({'prompt': 'test', 'stream': True})
        self.assertEqual(closed, [True])
        self.assertTrue(handler.close_connection)

    def test_nonstream_http_disconnect_sets_generation_cancellation(self):
        import http.client
        admitted, cancelled = threading.Event(), threading.Event()
        def generate(*args, cancel_event=None, **kwargs):
            admitted.set()
            if cancel_event.wait(3):
                cancelled.set()
            return 'cancelled', 0
        engine = SimpleNamespace(model_name='test', tok=SimpleNamespace(encode=lambda _: [1]),
                                 generate_text=generate, last_decode_tok_s=0, last_finish_reason='cancelled')
        with patch.object(serve, 'ENGINE', engine), patch.object(serve, 'STATS_TRACKER', None):
            server = ThreadingHTTPServer(('127.0.0.1', 0), serve.Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            connection = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=3)
            try:
                connection.request('POST', '/v1/completions', body=json.dumps({'prompt':'own test','max_tokens':65536}),
                                   headers={'Content-Type':'application/json'})
                self.assertTrue(admitted.wait(2))
                connection.close()
                self.assertTrue(cancelled.wait(2))
            finally:
                connection.close()
                server.shutdown()
                server.server_close()
                thread.join(3)

    def test_completion_and_jev_streams_have_http_eof(self):
        class FakeEngine:
            model_name = 'test'
            last_finish_reason = 'stop'
            last_decode_tok_s = 1
            tok = SimpleNamespace(encode=lambda *args, **kw: [1])
            @staticmethod
            def stream_text(*args, **kw):
                yield 'hello', None
                yield 'hello', 1
            @staticmethod
            def jev_inference(*args, **kw):
                return {'ok': True}, {'completion_tokens': 1, 'prompt_tokens': 1}
        with patch.object(serve, 'ENGINE', FakeEngine()), patch.object(serve, 'STATS_TRACKER', None):
            server = ThreadingHTTPServer(('127.0.0.1', 0), serve.Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            opener = build_opener(ProxyHandler({}))
            try:
                for body in ({'prompt': 'test', 'stream': True},
                             {'prompt': 'test', 'stream': True, 'jev': True, 'schema': {'ok': {'type': 'boolean'}}}):
                    req = Request(f'http://127.0.0.1:{server.server_port}/v1/completions',
                                  data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
                    with opener.open(req, timeout=3) as response:
                        self.assertEqual(response.headers['Connection'], 'close')
                        self.assertIn(b'data: [DONE]', response.read())
            finally:
                server.shutdown()
                server.server_close()
                thread.join(3)

    def test_mtp_metrics_are_shared_and_limit_is_released_on_header_failure(self):
        engine = SimpleNamespace(get_cache_stats=lambda: {'mtp': {'steps': 2}})
        with patch.object(serve, 'ENGINE', engine), patch.object(serve, '_MTP_CACHE', (None, 0, {})), patch.object(engine, 'get_cache_stats', wraps=engine.get_cache_stats) as query:
            self.assertEqual(serve.shared_mtp_stats(), {'steps': 2})
            self.assertEqual(serve.shared_mtp_stats(), {'steps': 2})
            self.assertEqual(query.call_count, 1)
        limit = threading.BoundedSemaphore(1)
        handler = serve.Handler.__new__(serve.Handler)
        handler._mtp_stream_body = lambda: (_ for _ in ()).throw(BrokenPipeError())
        with patch.object(serve, '_MTP_LIMIT', limit):
            with self.assertRaises(BrokenPipeError):
                handler._mtp_stream()
            self.assertTrue(limit.acquire(blocking=False))
            limit.release()


if __name__ == '__main__':
    unittest.main()
