"""Heartbeat ordering, disconnect cancellation and all generation endpoints."""
import io
import os
import threading
import types
import unittest
from unittest.mock import patch

from .sse import SSEWriter
from . import serve


class HeartbeatOutput(io.BytesIO):
    def __init__(self):
        super().__init__()
        self.beat = threading.Event()

    def write(self, data):
        count = super().write(data)
        if data.startswith(b':'):
            self.beat.set()
        return count


class SSETests(unittest.TestCase):
    def test_done_is_last_and_thread_stops(self):
        output = HeartbeatOutput()
        writer = SSEWriter(output, interval=.01).start()
        try:
            self.assertTrue(output.beat.wait(2))
            writer.write(b'data: {"ok":true}\n\n')
            writer.finish()
        finally:
            writer.close()
        self.assertFalse(writer.thread.is_alive())
        self.assertTrue(output.getvalue().endswith(b'data: [DONE]\n\n'))

    def test_broken_heartbeat_cancels_waiting_generation(self):
        cancel = threading.Event()
        output = types.SimpleNamespace(write=lambda _: (_ for _ in ()).throw(BrokenPipeError()))
        writer = SSEWriter(output, cancel, interval=.01).start()
        try:
            self.assertTrue(cancel.wait(2))
        finally:
            writer.close()
        self.assertFalse(writer.thread.is_alive())

    def test_disabled_heartbeat_has_no_thread(self):
        writer = SSEWriter(io.BytesIO(), interval=0).start()
        writer.finish()
        writer.close()
        self.assertIsNone(writer.thread)

    def test_iterator_creation_error_stops_heartbeat_and_ends_stream(self):
        handler = serve.Handler.__new__(serve.Handler)
        handler.wfile = io.BytesIO()
        handler.send_response = lambda *a: None
        handler.send_header = lambda *a: None
        handler.end_headers = lambda: None
        def fail(*a, **kw):
            raise RuntimeError('own setup failure')
        engine = types.SimpleNamespace(model_name='test', tok=types.SimpleNamespace(encode=lambda _: [1]),
                                       stream_text=fail)
        writers = []
        def make(*a, **kw):
            writers.append(SSEWriter(*a, **kw))
            return writers[-1]
        with patch.object(serve, 'ENGINE', engine), patch.object(serve, 'STATS_TRACKER', None), \
                patch.object(serve, 'SSEWriter', side_effect=make):
            handler._completion({'prompt': 'test', 'stream': True})
        self.assertFalse(writers[0].thread.is_alive())
        self.assertIn(b'generation_error', handler.wfile.getvalue())
        self.assertTrue(handler.wfile.getvalue().endswith(b'data: [DONE]\n\n'))

    @patch.dict(os.environ, {'DSV41_HTTP_HEARTBEAT_S': '.01'})
    def test_all_generation_streams_heartbeat_before_results(self):
        for endpoint in ('_chat', '_completion', '_jev'):
            with self.subTest(endpoint=endpoint):
                handler = serve.Handler.__new__(serve.Handler)
                handler.wfile = HeartbeatOutput()
                handler.headers = {}
                handler.path = '/v1/chat/completions' if endpoint == '_chat' else '/v1/completions'
                handler.send_response = lambda *a: None
                handler.send_header = lambda *a: None
                handler.end_headers = lambda: None
                def stream(*args, **kw):
                    self.assertTrue(handler.wfile.beat.wait(2))
                    yield 'hello', None
                    yield 'hello', 1
                def jev(*args, **kw):
                    self.assertTrue(handler.wfile.beat.wait(2))
                    return {'ok': True}, {'completion_tokens': 1, 'prompt_tokens': 1}
                engine = types.SimpleNamespace(
                    model_name='test', max_seq_len=100, last_decode_tok_s=1, last_finish_reason='stop',
                    tok=types.SimpleNamespace(encode=lambda *a, **kw: [1]), stream_text=stream,
                    format_chat=lambda *a, **kw: ([1], None, None),
                    parse_completion=lambda *a: {'content': 'hello'}, jev_inference=jev)
                with patch.object(serve, 'ENGINE', engine), patch.object(serve, 'STATS_TRACKER', None):
                    getattr(handler, endpoint)({'prompt': 'test', 'messages': [{'role': 'user', 'content': 'test'}],
                                               'schema': {'ok': {'type': 'boolean'}}, 'stream': True})
                payload = handler.wfile.getvalue()
                self.assertIn(b': keep-alive\n\n', payload)
                self.assertTrue(payload.endswith(b'data: [DONE]\n\n'))


if __name__ == '__main__':
    unittest.main()
