"""The legacy completion endpoint must use batch streaming and actual token counts."""
import io
import json
import types
import unittest
from unittest.mock import patch

from . import serve


class CompletionBatchStreamTests(unittest.TestCase):
    def test_final_tail_and_token_usage_are_preserved(self):
        engine = types.SimpleNamespace(
            model_name='deployed-model', tok=types.SimpleNamespace(encode=lambda text: [1, 2]),
            last_finish_reason='length', last_decode_tok_s=3,
            stream_text=lambda *a, **kw: iter([('中文', None), ('中文🌍', 7)]),
        )
        handler = serve.Handler.__new__(serve.Handler)
        handler.wfile = io.BytesIO()
        handler.send_response = lambda *a: None
        handler.send_header = lambda *a: None
        handler.end_headers = lambda: None
        with patch.object(serve, 'ENGINE', engine), patch.object(serve, 'STATS_TRACKER', None):
            handler._completion({'prompt': 'own test', 'stream': True, 'max_tokens': 7})
        lines = handler.wfile.getvalue().decode().splitlines()
        events = [json.loads(line[6:]) for line in lines if line.startswith('data: {')]
        self.assertEqual(''.join(e['choices'][0]['text'] for e in events), '中文🌍')
        self.assertEqual(events[-1]['usage']['completion_tokens'], 7)
        self.assertEqual(events[-1]['choices'][0]['finish_reason'], 'length')
        self.assertIn('data: [DONE]', lines)


if __name__ == '__main__':
    unittest.main()
