"""GPU-tier policy and complete checkpoint restoration without loading a model."""
import os
import threading
import types
import unittest
from unittest.mock import patch, Mock

import torch

from . import gpu_prefix_cache as cache
from . import test_batch_prefix_cache as fixtures


class GpuPrefixTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {
            'DSV41_GPU_PREFIX_CACHE': '1', 'DSV41_GPU_PREFIX_CACHE_GB': '4',
            'DSV41_GPU_PREFIX_RESERVE_GB': '8', 'DSV41_GPU_PREFIX_CACHE_ENTRIES': '4',
            'DSV41_DISABLE_PREFIX_CACHE': '0',
        })
        self.env.start()
        self.addCleanup(self.env.stop)
        self.e = fixtures.BatchPrefixCacheTests().engine()
        self.e.mtp = 4
        self.e._slot_tokens = {}
        self.e._slot_tokens_lock = threading.Lock()
        self.entries = []
        self.e._prefix_cache_entries = lambda: self.entries
        self.e.stats_tracker = None

    def entry(self, base=4, gpu=False, image_key=None):
        snap, size = self.e._snapshot_prefix_state(used_tokens=8)
        ent = dict(base_ids=list(range(base)), prompt_ids=list(range(base + 3)),
                   snapshot=snap, bytes=size, last_used=float(base), image_key=image_key)
        if gpu:
            ent['gpu_snapshot'] = [(k, h, key, s.clone()) for k, h, key, s in snap]
            ent['gpu_bytes'] = size
        self.entries.append(ent)
        return ent

    def promote_on_cpu(self, entry):
        # Exercise actual clone/allocation/commit logic; mock only CUDA transport.
        with patch.object(cache, '_target_device', return_value=torch.device('cuda:0')), \
             patch.object(cache, '_available', return_value=32 * 2**30), \
             patch.object(torch.Tensor, 'to', lambda t, **kw: t.clone()), \
             patch.object(torch.cuda, 'synchronize'):
            cache.promote(self.entries, entry)

    def test_gpu_first_then_cpu_then_miss_and_image_identity(self):
        short = self.entry(4, gpu=True)
        long = self.entry(7)
        self.assertIs(self.e._find_prefix_cache_entry(list(range(10))), short)
        cache.drop(short)
        self.assertIs(self.e._find_prefix_cache_entry(list(range(10))), long)
        self.assertIsNone(self.e._find_prefix_cache_entry([999, 1, 2]))
        self.assertIsNone(self.e._find_prefix_cache_entry(list(range(10)), image_key='different-image'))
        # Never rewind from an anchor at or after the input end.
        self.assertIs(self.e._find_prefix_cache_entry(list(range(7))), short)
        self.assertIsNone(self.e._find_prefix_cache_entry(list(range(4))))

    def test_checkpoint_survives_speculative_writes_and_restores_only_scratch(self):
        ent = self.entry(gpu=True)
        expected = [t[:1].clone() for t in self.e.all_tensors]
        for t in self.e.all_tensors:
            t.add_(700)  # rejected future rows, compressor/ring and draft pollution
        active = [t[1:].clone() for t in self.e.all_tensors]
        self.assertEqual(self.e._restore_prefix_entry(ent, 10), 'gpu_snapshot')
        for t, old, live in zip(self.e.all_tensors, expected, active):
            self.assertTrue(torch.equal(t[:1], old))
            self.assertTrue(torch.equal(t[1:], live))

    def test_promotion_is_independent_of_both_cpu_and_live_state(self):
        ent = self.entry()
        self.promote_on_cpu(ent)
        frozen = [s.clone() for _, _, _, s in ent['gpu_snapshot']]
        with torch.inference_mode():
            for _, _, _, s in ent['snapshot']:
                s.add_(200)
        for t in self.e.all_tensors:
            t.add_(400)
        for (_, _, _, actual), expected in zip(ent['gpu_snapshot'], frozen):
            self.assertTrue(torch.equal(actual, expected))

    def test_lru_evicts_gpu_only_and_preserves_cpu_fallback(self):
        old = self.entry(3, gpu=True)
        new = self.entry(5, gpu=True)
        with patch.dict(os.environ, {'DSV41_GPU_PREFIX_CACHE_ENTRIES': '1'}):
            self.assertTrue(cache.trim(self.entries))
        self.assertNotIn('gpu_snapshot', old)
        self.assertIn('snapshot', old)
        self.assertIn('gpu_snapshot', new)

    def test_pressure_and_oversized_entry_skip_promotion(self):
        ent = self.entry()
        with patch.object(cache, '_target_device', return_value=torch.device('cuda:0')), \
             patch.object(cache, '_available', return_value=1024):
            cache.promote(self.entries, ent)
        self.assertNotIn('gpu_snapshot', ent)
        with patch.dict(os.environ, {'DSV41_GPU_PREFIX_CACHE_GB': '0.000000001'}):
            self.promote_on_cpu(ent)
        self.assertNotIn('gpu_snapshot', ent)

    def test_pressure_evicts_resident_gpu_but_keeps_cpu_checkpoint(self):
        ent = self.entry(gpu=True)
        with patch.object(cache, '_available', return_value=1024):
            self.assertFalse(cache.trim(self.entries, needed={torch.device('cuda:0'): 1}))
        self.assertNotIn('gpu_snapshot', ent)
        self.assertIn('snapshot', ent)

    def test_partial_allocation_failure_does_not_publish_checkpoint(self):
        ent = self.entry()
        first = ent['snapshot'][0][3].clone()
        with patch.object(cache, '_target_device', return_value=torch.device('cuda:0')), \
             patch.object(cache, '_available', return_value=32 * 2**30), \
             patch.object(torch.Tensor, 'to', side_effect=[first, torch.OutOfMemoryError('test')]):
            cache.promote(self.entries, ent)
        self.assertNotIn('gpu_snapshot', ent)
        self.assertIn('snapshot', ent)

    def test_restore_failure_overwrites_partial_state_from_cpu(self):
        ent = self.entry(gpu=True)
        restore = self.e._restore_prefix_state
        calls = []
        def flaky(snapshot):
            calls.append(snapshot)
            if len(calls) == 1:
                self.e.all_tensors[0][0].fill_(-900)
                raise RuntimeError('synthetic GPU copy failure')
            restore(snapshot)
        with patch.object(self.e, '_restore_prefix_state', side_effect=flaky):
            self.assertEqual(self.e._restore_prefix_entry(ent, 10), 'host_replay')
        self.assertNotIn('gpu_snapshot', ent)
        self.assertTrue(torch.equal(self.e.all_tensors[0][:1], ent['snapshot'][0][3]))

    def test_disabled_switch_uses_cpu_and_single_mode_does_not_promote(self):
        ent = self.entry(gpu=True)
        with patch.dict(os.environ, {'DSV41_GPU_PREFIX_CACHE': '0'}):
            self.assertEqual(self.e._restore_prefix_entry(ent, 10), 'host_replay')
        self.e.max_seqs = 1
        self.assertFalse(cache.enabled(self.e))

    def test_batch_mtp_prefill_uses_gpu_checkpoint_without_live_slot_reuse(self):
        ent = self.entry(gpu=True)
        self.e._slot_tokens = {1: list(range(20))}
        self.e._prefill_with_gpu_slot_reuse = Mock(side_effect=AssertionError('unsafe live slot'))
        self.e._replay_prefix_tail = Mock(return_value=(torch.ones(1, 2), None))
        self.e._record_prefill_stats = Mock()
        self.e._promote_prefix_entry = Mock()
        _, reused = self.e._prefill_with_prefix_reuse(list(range(10)))
        self.assertEqual(reused, 4)
        self.assertEqual(self.e._record_prefill_stats.call_args.kwargs['prefill_type'], 'gpu_snapshot')
        self.e._replay_prefix_tail.assert_called_once_with(list(range(10)), 4, snapshot_at=None)
        self.e._prefill_with_gpu_slot_reuse.assert_not_called()
        self.assertEqual(ent['snapshot'][0][3].device.type, 'cpu')

    @unittest.skipUnless(os.environ.get('DSV41_TEST_GPU_PREFIX_DEVICE'), 'explicit GPU smoke opt-in')
    def test_real_gpu_transport_and_restoration(self):
        device = os.environ['DSV41_TEST_GPU_PREFIX_DEVICE']
        from .engine import Engine
        e = Engine.__new__(Engine)
        e.max_seqs, e.mtp = 3, 4
        e.model = types.SimpleNamespace(shared=None)
        owners = [types.SimpleNamespace(cache=torch.arange(3 * 128 * 8, device=device).reshape(3, 128, 8).float())
                  for _ in range(4)]
        e._prefix_cache_slots = lambda: [('attr', h, 'cache', h.cache[:1]) for h in owners]
        snap, size = e._snapshot_prefix_state(used_tokens=128)
        ent = dict(base_ids=list(range(128)), snapshot=snap, bytes=size, last_used=0)
        cache.promote([ent], ent)
        self.assertTrue(all(t.is_cuda for _, _, _, t in ent['gpu_snapshot']))
        expected = [h.cache[:1].clone() for h in owners]
        for h in owners:
            h.cache.add_(10000)
        live = [h.cache[1:].clone() for h in owners]
        self.assertEqual(e._restore_prefix_entry(ent, 140), 'gpu_snapshot')
        for h, old, active in zip(owners, expected, live):
            self.assertTrue(torch.equal(h.cache[:1], old))
            self.assertTrue(torch.equal(h.cache[1:], active))
        cache.drop(ent)


if __name__ == '__main__':
    unittest.main()
