"""Streaming/final parser agreement using the checkpoint codec, without weights."""
import contextlib
import io
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from dsv41.engine import Engine
from dsv41.streaming import ChatStreamSplitter
from dsv41 import serve


CALLS = ('<｜DSML｜ calls>\n<｜DSML｜ invoke name="shell">\n'
         '<｜DSML｜ parameter name="command" string="true">123</｜DSML｜ parameter>\n'
         '</｜DSML｜ invoke>\n</｜DSML｜ calls>')


class ChatParseTests(unittest.TestCase):
    def setUp(self):
        codec = Path('/models/DeepSeek-V4.1-Flash/encoding')
        if not codec.exists():
            self.skipTest('checkpoint encoding package not installed')
        sys.path.insert(0, str(codec))
        from encoding import eos_token, parse_message_from_completion_text
        self.engine = Engine.__new__(Engine)
        self.engine.tok = SimpleNamespace(eos_token=eos_token)
        self.engine.thinking_mode = 'chat'
        self.engine._parse = Mock(wraps=parse_message_from_completion_text)
        self.engine._fallback_parse_dsml = Mock(wraps=Engine._fallback_parse_dsml)

    def check_agreement(self, raw, mode, expected_reasoning, expected_content):
        final = self.engine.parse_completion(raw, mode)
        self.assertEqual(final.get('reasoning_content') or '', expected_reasoning)
        self.assertEqual(final.get('content') or '', expected_content)
        # Every possible two-piece boundary includes splits inside control tags.
        for index in range(len(raw) + 1):
            splitter = ChatStreamSplitter(thinking=mode == 'thinking')
            deltas = splitter.push(raw[:index]) + splitter.push(raw[index:])
            for key, emitted in (('reasoning_content', splitter.reasoning), ('content', splitter.content)):
                expected = final.get(key) or ''
                with self.subTest(mode=mode, boundary=index, field=key):
                    self.assertTrue(expected.startswith(emitted), (repr(expected), repr(emitted)))
                    self.assertEqual(''.join(d.get(key, '') for d in deltas), emitted)
        splitter = ChatStreamSplitter(thinking=mode == 'thinking')
        for char in raw:
            splitter.push(char)
        self.assertTrue(expected_reasoning.startswith(splitter.reasoning))
        self.assertTrue(expected_content.startswith(splitter.content))
        return final

    def test_tool_delimiter_is_not_visible_content(self):
        for mode in ('chat', 'thinking'):
            for content in ('', '正文', '  正文  '):
                reasoning = '  推理\n\n' if mode == 'thinking' else ''
                raw = (reasoning + '</think>' if reasoning else '') + content + '\n\n' + CALLS
                with self.subTest(mode=mode, content=content):
                    final = self.check_agreement(raw, mode, reasoning, content)
                    self.assertEqual(json.loads(final['tool_calls'][0]['function']['arguments']), {'command': '123'})
        self.engine._fallback_parse_dsml.assert_not_called()
        self.assertTrue(all(c.kwargs['thinking_mode'] == 'chat' for c in self.engine._parse.call_args_list))

    def test_whitespace_is_preserved_without_tools(self):
        self.check_agreement('  正文\n\n', 'chat', '', '  正文\n\n')
        self.check_agreement('  推理\n\n</think>  正文\n\n', 'thinking', '  推理\n\n', '  正文\n\n')
        self.check_agreement('<think>  推理\n</think>正文', 'thinking', '  推理\n', '正文')
        self.check_agreement('  未完成推理\n\n', 'thinking', '  未完成推理\n\n', '')
        self.check_agreement('<think>  未完成推理\n', 'thinking', '  未完成推理\n', '')
        self.engine._fallback_parse_dsml.assert_not_called()

    def test_tool_call_without_closing_think(self):
        self.check_agreement('  推理\n\n' + CALLS, 'thinking', '  推理', '')
        self.check_agreement('<think>  推理\n\n' + CALLS, 'thinking', '  推理', '')

    def test_fallback_preserves_whitespace(self):
        self.engine._parse.side_effect = ValueError('malformed completion')
        self.check_agreement('  推理\n</think>  正文  \n\n' + CALLS,
                             'thinking', '  推理\n', '  正文  ')
        self.check_agreement('  正文\n\n', 'chat', '', '  正文\n\n')
        self.engine._fallback_parse_dsml.assert_called_once()

    def test_chat_sse_agrees_with_final_parse_and_has_no_warning(self):
        for content in ('', '正文', '  正文\n\n'):
            # Also exercise flushing held final newlines when there is no tool.
            raw = '  推理\n</think>' + content
            if not content.endswith('\n\n'):
                raw += '\n\n' + CALLS
            expected = self.engine.parse_completion(raw, 'thinking')
            self.engine.model_name = 'test'
            self.engine.max_seq_len = 4096
            self.engine.last_decode_tok_s = 1
            self.engine.last_finish_reason = 'stop'
            self.engine.format_chat = lambda *a, **kw: ([1], None, None)
            def stream(*a, **kw):
                for char in raw:
                    yield char, None
                yield raw, len(raw)
            self.engine.stream_text = stream
            handler = serve.Handler.__new__(serve.Handler)
            handler.headers = {}
            handler.wfile = io.BytesIO()
            handler.send_response = lambda *a: None
            handler.send_header = lambda *a: None
            handler.end_headers = lambda: None
            log = io.StringIO()
            with patch.object(serve, 'ENGINE', self.engine), patch.object(serve, 'STATS_TRACKER', None), contextlib.redirect_stdout(log):
                handler._chat({'stream': True, 'thinking': True, 'messages': [{'role': 'user', 'content': 'test'}]})
            self.assertNotIn('diverged', log.getvalue())
            chunks = [json.loads(line[6:]) for line in handler.wfile.getvalue().decode().splitlines()
                      if line.startswith('data: ') and line != 'data: [DONE]']
            deltas = [c['choices'][0]['delta'] for c in chunks]
            for field in ('content', 'reasoning_content'):
                self.assertEqual(''.join(d.get(field, '') for d in deltas), expected.get(field) or '')
            self.assertEqual(chunks[-1]['choices'][0]['finish_reason'], 'tool_calls' if expected.get('tool_calls') else 'stop')
            self.assertTrue(handler.wfile.getvalue().endswith(b'data: [DONE]\n\n'))


if __name__ == '__main__':
    unittest.main()
