"""Prefix snapshots must restore scratch slot 0 without touching live decode slots."""
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import torch

from .engine import Engine


class Compressor:
    pass


class NgramHashState:
    pass


class SharedAttn:
    pass


class BatchPrefixCacheTests(unittest.TestCase):
    def engine(self, slots=3):
        e = Engine.__new__(Engine)
        e.max_seqs, e.max_seq_len = slots, 1048576
        e.tok = types.SimpleNamespace(name_or_path='test-batch-cache')
        def tensor():
            return torch.arange(slots * 16).reshape(slots, 8, 2).float()
        compressor = Compressor()
        for name in ('kv_state', 'score_state', 'kv_ring', 'score_ring'):
            setattr(compressor, name, tensor())
        shared = SharedAttn()
        shared.compress_kv = {(0, 'cpu'): tensor(), (0, 'mirror'): tensor()}
        shared.index_k = {(0, 'cpu'): tensor(), (0, 'mirror'): tensor()}
        shared.cache_max_rows = {0: 8}
        shared.kv_owner, shared.index_owner = 0, 0
        engram = NgramHashState()
        engram.cache = tensor()
        attn = types.SimpleNamespace(window_kv_cache=tensor(), compressor=compressor)
        e.model = types.SimpleNamespace(blocks=[types.SimpleNamespace(attn=attn)], shared=shared,
                                        engram_hash=engram, args=types.SimpleNamespace(compress_ratios=[1]))
        e.ds = types.SimpleNamespace(blocks=[types.SimpleNamespace(attn=types.SimpleNamespace(window_kv_cache=tensor()))])
        e.all_tensors = [attn.window_kv_cache, engram.cache, e.ds.blocks[0].attn.window_kv_cache,
                         *shared.compress_kv.values(), *shared.index_k.values(),
                         *(getattr(compressor, n) for n in ('kv_state', 'score_state', 'kv_ring', 'score_ring'))]
        return e

    def test_snapshot_and_restore_isolate_all_sequence_state_and_mirrors(self):
        e = self.engine()
        expected = [t[0:1].clone() for t in e.all_tensors]
        snap, _ = e._snapshot_prefix_state(used_tokens=8)
        for kind, _, _, saved in snap:
            if kind != 'scalar_attr':
                self.assertEqual(saved.shape[0], 1)
        for t in e.all_tensors:
            t[0].fill_(-1)
            t[1:].add_(1000)
        other_slots = [t[1:].clone() for t in e.all_tensors]
        e._restore_prefix_state(snap)
        for t, first, others in zip(e.all_tensors, expected, other_slots):
            self.assertTrue(torch.equal(t[:1], first))
            self.assertTrue(torch.equal(t[1:], others))

    def test_scalar_replay_explicitly_targets_only_scratch_slot(self):
        e = self.engine()
        e.mtp = 0
        calls = []
        e.rt = types.SimpleNamespace(step=lambda *a, **kw: self.fail('full batch runtime used'))
        def step(tokens, positions, **kwargs):
            calls.append((tokens, positions, kwargs))
            return torch.zeros(1, 10)
        e.rt_b1 = types.SimpleNamespace(step=step)
        with patch.dict(os.environ, {'DSV41_PREFIX_BLOCK_REPLAY': '0'}):
            e._replay_prefix_tail([1, 2, 3], 1)
        self.assertEqual(calls, [([2], [1], {'seq': [0], 'pmax': [1]}),
                                 ([3], [2], {'seq': [0], 'pmax': [2]})])

    def test_metrics_do_not_load_snapshots_or_grow_gpu_caches(self):
        e = self.engine()
        e._prefix_cache_entries = lambda: self.fail('metrics attempted snapshot import')
        self.assertEqual(e._prefix_cache_stats(), (0, 0))
        e._rolling_prefix_entries = [{'bytes':100}, {'bytes':200}]
        self.assertEqual(e._prefix_cache_stats(), (2, 300))

    def test_legacy_multislot_snapshot_rejected_before_any_write(self):
        e = self.engine()
        snap, _ = e._snapshot_prefix_state()
        kind, holder, key, src = snap[-1]
        # Append an unsafe tensor after otherwise valid entries.
        snap.append(('attr', e.model.blocks[0].attn, 'window_kv_cache', torch.ones(3, 8, 2)))
        for t in e.all_tensors: t.add_(900)
        before = [t.clone() for t in e.all_tensors]
        with self.assertRaisesRegex(ValueError, 'slot 0'):
            e._restore_prefix_state(snap)
        for actual, expected in zip(e.all_tensors, before):
            self.assertTrue(torch.equal(actual, expected))

    def test_persisted_text_and_image_snapshots_can_move_between_slot_counts(self):
        for image_key in (None, 'own-test-pixel-digest'):
            with self.subTest(image_key=image_key), tempfile.TemporaryDirectory() as tmp:
                with patch.dict(os.environ, {'DSV41_PREFIX_CACHE_DIR': tmp, 'DSV41_IMAGE_PREFIX_CACHE_DIR': tmp}):
                    source = self.engine(slots=1)
                    snap, size = source._snapshot_prefix_state(used_tokens=8)
                    source._persist_prefix_cache_entry(dict(prompt_ids=list(range(10)), base_ids=list(range(8)),
                                                           snapshot=snap, bytes=size, image_key=image_key))
                    target = self.engine(slots=3)
                    for t in target.all_tensors: t.add_(300)
                    others = [t[1:].clone() for t in target.all_tensors]
                    loaded = []
                    target._load_prefix_cache_from_tmpfs(loaded, root=tmp)
                    self.assertEqual(len(loaded), 1)
                    target._restore_prefix_state(loaded[0]['snapshot'])
                    for actual, original, other in zip(target.all_tensors, source.all_tensors, others):
                        self.assertTrue(torch.equal(actual[:1], original[:1]))
                        self.assertTrue(torch.equal(actual[1:], other))


if __name__ == '__main__':
    unittest.main()
