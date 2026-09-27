"""Committed-state checks for mutable GPU slots and Jev scratch isolation."""
import collections
import os
import threading
import types
import unittest
from unittest.mock import Mock, patch

import torch

from .engine import Engine
from . import gpu_prefix_cache


class LiveSlotSafetyTests(unittest.TestCase):
    def engine(self, end=80):
        e = Engine.__new__(Engine)
        e.mtp = 0
        e._slot_tokens = {2: list(range(end))}
        e._slot_cache_ends = {2: end}
        e._slot_tokens_lock = threading.Lock()
        e.rt = types.SimpleNamespace(copy_seq=Mock())
        e._forward_prefix_continuation = Mock(return_value=torch.ones(1, 2))
        e.prefill_history = collections.deque(maxlen=10)
        e.stats_tracker = None
        return e

    @patch.dict(os.environ, {'DSV41_LIVE_SLOT_REUSE': '1'})
    def test_overwritten_history_is_refused_before_any_write(self):
        e = self.engine(end=1024)
        prompt = list(range(900)) + [-10] * 124
        self.assertEqual(e._find_best_gpu_slot(prompt), (-1, 0))
        with self.assertRaises(ValueError):
            e._prefill_with_gpu_slot_reuse(prompt, 2, 900)
        e.rt.copy_seq.assert_not_called()
        e._forward_prefix_continuation.assert_not_called()

    @patch.dict(os.environ, {'DSV41_LIVE_SLOT_REUSE': '1'})
    def test_exact_continuation_matches_cold_window_computation(self):
        e = self.engine()
        ring = torch.zeros(3, 8)
        for pos in range(80):
            ring[2, pos % 8] = pos
        untouched = ring[1:].clone()
        def copy(src, dst, **kw):
            ring[dst].copy_(ring[src])
        def forward(ids, start, **kw):
            scores = []
            for pos in range(start, len(ids)):
                history = [ring[0, p % 8].item() for p in range(max(0, pos - 7), pos)]
                scores.append(sum(history) + ids[pos])
                ring[0, pos % 8] = ids[pos]
            return torch.tensor([scores])
        e.rt.copy_seq = copy
        e._forward_prefix_continuation = forward
        prompt = list(range(100))
        actual, reused = e._prefill_with_gpu_slot_reuse(prompt, 2, 80)
        expected = torch.tensor([[sum(prompt[max(0, p - 7):p + 1]) for p in range(80, 100)]], dtype=actual.dtype)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(ring[1:], untouched)
        self.assertEqual(reused, 80)
        self.assertNotIn(0, e._slot_cache_ends)

    @patch.dict(os.environ, {'DSV41_LIVE_SLOT_REUSE': '1'})
    def test_no_rewind_unknown_state_or_speculative_reuse(self):
        for end, lcp, total in ((80, 80, 80), (81, 100, 110), (96, 80, 110)):
            e = self.engine(end)
            self.assertIsNone(e._gpu_slot_reuse_pos(2, lcp, total))
        e = self.engine()
        e._invalidate_live_slot(2)
        self.assertIsNone(e._gpu_slot_reuse_pos(2, 80, 100))
        e._mark_live_slot(2, 80)
        e.mtp = 4
        self.assertIsNone(e._gpu_slot_reuse_pos(2, 80, 100))

    def test_live_switch_is_independent_of_gpu_checkpoints(self):
        e = self.engine()
        e.max_seqs, e.mtp = 3, 4
        with patch.dict(os.environ, {'DSV41_LIVE_SLOT_REUSE': '0', 'DSV41_GPU_PREFIX_CACHE': '1',
                                     'DSV41_GPU_PREFIX_CACHE_GB': '4'}):
            self.assertEqual(e._find_best_gpu_slot(list(range(100))), (-1, 0))
            self.assertTrue(gpu_prefix_cache.enabled(e))
        e.mtp = 0
        with patch.dict(os.environ, {'DSV41_LIVE_SLOT_REUSE': '1', 'DSV41_GPU_PREFIX_CACHE': '0'}):
            self.assertEqual(e._find_best_gpu_slot(list(range(100))), (2, 80))

    def test_jev_uses_only_scratch_and_invalidates_it_on_success_and_failure(self):
        for failure in (False, True):
            with self.subTest(failure=failure):
                e = self.engine()
                e.lock = threading.Lock()
                e.current_phase = 'decode'
                e._slot_tokens[0] = list(range(80))
                e._slot_cache_ends[0] = 80
                e._prefix_prompt_ids, e._prefix_len = [1, 2], 2
                e._record_prefill_stats = Mock()
                ring = torch.arange(24).reshape(3, 8).clone()
                others = ring[1:].clone()
                def process(prompt, schema, *, max_batch):
                    self.assertEqual(max_batch, 1)
                    ring[:max_batch].fill_(-1)
                    if failure:
                        raise RuntimeError('own test failure')
                    return {'ok': True}, {}
                e._jev_engine = types.SimpleNamespace(process_request=process)
                if failure:
                    with self.assertRaises(RuntimeError):
                        e.jev_inference('test', {}, max_batch=32)
                else:
                    e.jev_inference('test', {}, max_batch=32)
                torch.testing.assert_close(ring[1:], others)
                self.assertNotIn(0, e._slot_tokens)
                self.assertNotIn(0, e._slot_cache_ends)
                self.assertEqual(e._slot_cache_ends[2], 80)
                self.assertIsNone(e._prefix_prompt_ids)
                self.assertEqual(e._prefix_len, 0)
                self.assertEqual(e.current_phase, 'decode')


if __name__ == '__main__':
    unittest.main()
