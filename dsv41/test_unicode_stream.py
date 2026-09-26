"""Regression checks for split UTF-8 characters in streamed token text."""
import unittest
import threading
from pathlib import Path

from dsv41.streaming import IncrementalTokenDecoder


class BytesTokenizer:
    def decode(self, token_ids, errors='replace'):
        return bytes(token_ids).decode('utf-8', errors=errors)


class UnicodeStreamTests(unittest.TestCase):
    def test_waits_for_complete_multibyte_character(self):
        decoder = IncrementalTokenDecoder(BytesTokenizer())
        pieces = [decoder.push(byte) for byte in '🌍 国际'.encode('utf-8')]
        self.assertEqual(pieces[:3], ['', '', ''])
        self.assertEqual(''.join(pieces) + decoder.flush(), '🌍 国际')
        self.assertNotIn('\ufffd', ''.join(pieces))

    def test_incomplete_terminal_sequence_is_flushed(self):
        decoder = IncrementalTokenDecoder(BytesTokenizer())
        self.assertEqual(decoder.push(0xf0), '')
        self.assertEqual(decoder.flush(), '\ufffd')

    def test_malformed_byte_does_not_hide_rest_of_stream(self):
        decoder = IncrementalTokenDecoder(BytesTokenizer())
        pieces = [decoder.push(byte) for byte in b'visible \xff' + b'more text after error']
        rendered = ''.join(pieces) + decoder.flush()
        self.assertEqual(rendered, 'visible \ufffdmore text after error')
        self.assertIn('more text after error', ''.join(pieces))

    def test_single_sequence_stream_reports_generated_tokens(self):
        from dsv41.engine import Engine

        class FakeEngine:
            max_seqs = 1
            _slot_lock = threading.Lock()
            slot_states = {0: {'generated_tokens': 3}}

            def generate(self, *args, **kwargs):
                yield 1, '中'
                yield 3, '文'

        result = list(Engine.stream_text(FakeEngine(), [], None))
        self.assertEqual(result, [('中', None), ('文', None), ('中文', 3)])

    def test_model_tokenizer_reproduces_user_heading(self):
        ckpt = Path('/models/DeepSeek-V4.1-Flash')
        if not ckpt.exists():
            self.skipTest('official checkpoint tokenizer not installed')
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(ckpt)
        heading = '🌍 国际 · 地缘冲突'
        ids = tokenizer.encode(heading, add_special_tokens=False)
        self.assertEqual(''.join(tokenizer.decode([token_id], errors='replace') for token_id in ids),
                         '�� 国际 · 地缘冲突')
        decoder = IncrementalTokenDecoder(tokenizer)
        pieces = [decoder.push(token_id) for token_id in ids]
        self.assertEqual(pieces[0], '')
        self.assertEqual(''.join(pieces) + decoder.flush(), heading)
        self.assertNotIn('\ufffd', ''.join(pieces))


if __name__ == '__main__':
    unittest.main()
