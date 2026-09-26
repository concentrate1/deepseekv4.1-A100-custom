"""Reasoning-effort API mapping and prompt encoding, without loading model weights."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

from dsv41.engine import Engine
from dsv41.serve import _reasoning_mode_and_effort


class ReasoningEffortTest(unittest.TestCase):
    def test_api_effort_mapping(self):
        expected = {
            'low': ('thinking', 'low'),
            'medium': ('thinking', 65),
            'high': ('thinking', 'high'),
            'max': ('thinking', 'max'),
            'minimal': ('thinking', 1),
            'xhigh': ('thinking', 90),
            'none': (None, None),
            20: ('thinking', 20),
            100: ('thinking', 100),
        }
        for value, result in expected.items():
            with self.subTest(value=value):
                self.assertEqual(_reasoning_mode_and_effort({'reasoning_effort': value}), result)
        self.assertEqual(_reasoning_mode_and_effort({'thinking': True}), ('thinking', None))
        self.assertEqual(_reasoning_mode_and_effort({}), (None, None))

    def test_invalid_effort_is_rejected(self):
        for value in (0, 101, True, 'extreme', 50.5):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    _reasoning_mode_and_effort({'reasoning_effort': value})

    def test_official_encoder_receives_distinct_numeric_effort(self):
        ckpt_encoding = Path('/models/DeepSeek-V4.1-Flash/encoding')
        if not ckpt_encoding.exists():
            self.skipTest('official checkpoint encoding package not installed')
        sys.path.insert(0, str(ckpt_encoding))
        from encoding import encode_messages
        engine = Engine.__new__(Engine)
        engine._encode = encode_messages
        engine.thinking_mode = 'chat'
        messages = [{'role': 'user', 'content': 'question'}]
        for value, budget in [('low', 50), ('medium', 65), ('high', 75), ('xhigh', 90), ('max', 100), (20, 20)]:
            mode, effort = _reasoning_mode_and_effort({'reasoning_effort': value})
            prompt = engine.chat_prompt(messages, mode, effort)
            with self.subTest(value=value):
                self.assertIn(f'Reasoning Effort: {budget} ', prompt)
                self.assertTrue(prompt.endswith('<think>'))
        mode, effort = _reasoning_mode_and_effort({'reasoning_effort': 'none'})
        prompt = engine.chat_prompt(messages, mode, effort)
        self.assertNotIn('Reasoning Effort:', prompt)

    def test_multimodal_format_path_forwards_effort(self):
        engine = Engine.__new__(Engine)
        calls = []
        def fake_encode(messages, **kwargs):
            calls.append(kwargs)
            return 'prompt', {'images': []}
        engine._encode = fake_encode
        engine.thinking_mode = 'chat'
        engine.tok = SimpleNamespace(encode=lambda value: [1, 2, 3])
        ids, images, token_types = engine.format_chat(
            [{'role': 'user', 'content': 'question'}], 'thinking', reasoning_effort=65)
        self.assertEqual(ids, [1, 2, 3])
        self.assertIsNone(images)
        self.assertIsNone(token_types)
        self.assertEqual(calls[0]['thinking_mode'], 'thinking')
        self.assertEqual(calls[0]['reasoning_effort'], 65)
        self.assertTrue(calls[0]['return_multi_modal_data'])


    def test_unclosed_thinking_is_not_visible_content(self):
        ckpt_encoding = Path('/models/DeepSeek-V4.1-Flash/encoding')
        if not ckpt_encoding.exists():
            self.skipTest('official checkpoint encoding package not installed')
        sys.path.insert(0, str(ckpt_encoding))
        from encoding import parse_message_from_completion_text
        engine = Engine.__new__(Engine)
        engine.tok = SimpleNamespace(eos_token='<｜end▁of▁sentence｜>')
        engine._parse = parse_message_from_completion_text
        engine.thinking_mode = 'chat'
        incomplete = engine.parse_completion('Need to calculate 1987 times 2033', 'thinking')
        self.assertEqual(incomplete['content'], '')
        self.assertEqual(incomplete['reasoning_content'], 'Need to calculate 1987 times 2033')
        complete = engine.parse_completion('Reasoning done.</think>Final answer', 'thinking')
        self.assertEqual(complete['content'], 'Final answer')
        self.assertEqual(complete['reasoning_content'], 'Reasoning done.')



if __name__ == '__main__':
    unittest.main()
