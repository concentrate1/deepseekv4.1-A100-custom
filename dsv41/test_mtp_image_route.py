"""Image inputs must reach the MTP prefill instead of silently using plain decode."""
import threading
import types
import unittest
from unittest.mock import Mock

import torch

from .engine import Engine, GenParams


class MTPImageRouteTest(unittest.TestCase):
    def test_generate_routes_image_inputs_to_mtp(self):
        engine = object.__new__(Engine)
        engine.max_seq_len = 1024
        engine.mtp = 5
        engine.ds = object()
        engine.model = types.SimpleNamespace(blocks=[types.SimpleNamespace(device=torch.device("cpu"))])
        engine.lock = threading.Lock()
        engine._slot_lock = threading.Lock()
        engine.slot_states = {0: {"generated_tokens": 1, "tok_s": 1.0}}
        engine.mtp_stats = {"status": "idle"}
        engine.last_finish_reason = "stop"
        engine.stats_tracker = None
        seen = {}

        def fake_mtp(self, ids, params, max_new, generator, images=None, token_types=None):
            seen.update(ids=ids, images=images, token_types=token_types)
            yield 42, "answer"

        engine._generate_mtp_locked = types.MethodType(fake_mtp, engine)
        images = [[object()]]
        token_types = torch.tensor([[-1, 1]], dtype=torch.long)
        result = list(engine.generate([10, 11], GenParams(max_new_tokens=3),
                                      images=images, token_types=token_types))
        self.assertEqual(result, [(42, "answer")])
        self.assertIs(seen["images"], images)
        self.assertIs(seen["token_types"], token_types)
        self.assertEqual(engine.mtp_stats["status"], "completed")

    def test_image_prefill_forwards_pixels_and_token_types(self):
        engine = object.__new__(Engine)
        images = [[object()]]
        token_types = torch.tensor([[-1, 1]], dtype=torch.long)
        captured = {}

        def forward(ids, start_pos, images=None, token_types=None):
            captured.update(ids=ids, start_pos=start_pos,
                            images=images, token_types=token_types)
            return torch.zeros((1, 10))

        engine.model = types.SimpleNamespace(
            blocks=[types.SimpleNamespace(device=torch.device("cpu"))],
            forward=forward,
        )
        engine._prefix_prompt_ids = [1]
        engine._prefix_len = 1
        engine._slot_tokens_lock = threading.Lock()
        engine._slot_tokens = {}
        engine._disable_prefix_cache_once = True
        engine._image_cache_identity = Mock(return_value=("image", 2))
        engine._record_prefill_stats = Mock()
        logits, reused = engine._prefill_with_prefix_reuse([10, 11], images, token_types)
        self.assertEqual(tuple(logits.shape), (1, 10))
        self.assertEqual(reused, 0)
        self.assertIs(captured["images"], images)
        self.assertIs(captured["token_types"], token_types)
        self.assertIsNone(engine._prefix_prompt_ids)
        self.assertEqual(engine._prefix_len, 0)


if __name__ == "__main__":
    unittest.main()
