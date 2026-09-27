"""Prefill handoffs preserve metadata, avoid cache growth, and never recurse or acquire the engine lock."""
import threading
import types
import unittest
from unittest.mock import Mock, patch

import torch
from .engine import Engine


class CooperativePrefillTests(unittest.TestCase):
    def engine(self):
        e = Engine.__new__(Engine)
        req = types.SimpleNamespace(pos=10, out_tokens=[1])
        e._active_slots = {1:req}
        e.mtp = 4
        e.max_seq_len = 1000000
        e.ds = types.SimpleNamespace(block=5)
        e.rt_mtp_single = types.SimpleNamespace(B=5, _graph_cache_signature=('current',),
            _cache_signature=lambda: ('current',), _prepare_decode_cache=Mock())
        e.model = types.SimpleNamespace(blocks=[types.SimpleNamespace(attn=types.SimpleNamespace(ratio=1))],
            shared=types.SimpleNamespace(_current_chunk_idx=7, kv_owner=2, index_owner=8,
                compress_kv={(0,'cpu'):torch.zeros(3,100,2)}, index_k={(0,'cpu'):torch.zeros(3,100,2)}))
        e.current_phase = 'prefill'
        e.prefill_interleave_stats = dict(slices=0,decode_steps=0,decode_tokens=0,deferred_for_cache=0)
        e._slot_lock = threading.Lock()
        e.mtp_stats = {'status':'prefilling'}
        e._sync_prefill_devices = Mock()
        e.lock = threading.Lock()
        return e, req

    def test_handoff_advances_existing_request_and_restores_prefill_metadata(self):
        e,req=self.engine()
        def decode(**kwargs):
            e.model.shared._current_chunk_idx=99
            e.model.shared.kv_owner=99
            e.model.shared.index_owner=99
            e.current_phase='decode'
            req.out_tokens.append(2)
            e._active_slots.clear()
        e._decode_active_once=Mock(side_effect=decode)
        # The scheduler already owns this non-reentrant lock.
        with e.lock:
            e._yield_prefill_decode()
        self.assertEqual(e.current_phase,'prefill')
        self.assertEqual((e.model.shared._current_chunk_idx,e.model.shared.kv_owner,e.model.shared.index_owner),(7,2,8))
        self.assertEqual(e.prefill_interleave_stats['decode_tokens'],1)
        self.assertEqual(e.prefill_interleave_stats['decode_steps'],1)
        self.assertEqual(e._sync_prefill_devices.call_count,2)
        self.assertFalse(e._cooperating_prefill)

    def test_exception_restores_metadata_and_reentry_guard(self):
        e,req=self.engine()
        def fail(**kwargs):
            e.current_phase='decode'
            e.model.shared.kv_owner=99
            raise RuntimeError('own test failure')
        e._decode_active_once=Mock(side_effect=fail)
        with self.assertRaisesRegex(RuntimeError,'own test'):
            e._yield_prefill_decode()
        self.assertEqual(e.current_phase,'prefill')
        self.assertEqual(e.model.shared.kv_owner,2)
        self.assertFalse(e._cooperating_prefill)
        self.assertEqual(e._sync_prefill_devices.call_count,2)

    def test_sync_failure_still_restores_python_state(self):
        e,req=self.engine()
        def decode(**kwargs):
            e.current_phase='decode'
            e.model.shared.index_owner=99
            e._active_slots.clear()
        e._decode_active_once=Mock(side_effect=decode)
        e._sync_prefill_devices.side_effect=[None,RuntimeError('sync failed')]
        with self.assertRaisesRegex(RuntimeError,'sync failed'):
            e._yield_prefill_decode()
        self.assertEqual(e.current_phase,'prefill')
        self.assertEqual(e.model.shared.index_owner,8)
        self.assertFalse(e._cooperating_prefill)

    def test_context_tail_does_not_switch_to_scratch_using_runtime(self):
        e,req=self.engine()
        e.max_seq_len=14
        e._decode_active_once=Mock()
        e._yield_prefill_decode()
        e._decode_active_once.assert_not_called()

    def test_capture_is_prepared_outside_live_pipeline_activations(self):
        e,req=self.engine()
        e._prepare_prefill_decode()
        e.rt_mtp_single._prepare_decode_cache.assert_called_once_with(15)

    def test_prepare_reserves_the_whole_new_prefill(self):
        e,req=self.engine()
        e._prepare_prefill_decode(end_pos=500)
        e.rt_mtp_single._prepare_decode_cache.assert_called_once_with(500)

    def test_layer_interval_skips_too_frequent_handoffs(self):
        e,req=self.engine()
        e._prefill_last_decode_yield=100.0
        e._decode_active_once=Mock()
        with patch('dsv41.engine.time.perf_counter',return_value=100.1):
            self.assertFalse(e._yield_prefill_decode(min_interval_s=.25))
        e._decode_active_once.assert_not_called()
        e._sync_prefill_devices.assert_not_called()
        self.assertEqual(e.prefill_interleave_stats['slices'],0)

    def test_layer_handoff_reports_real_work_and_resets_timer(self):
        e,req=self.engine()
        e._prefill_last_decode_yield=99.0
        e._decode_active_once=Mock(side_effect=lambda **kw: e._active_slots.clear())
        with patch('dsv41.engine.time.perf_counter',return_value=100.0):
            self.assertTrue(e._yield_prefill_decode(min_interval_s=.25))
        self.assertEqual(e._prefill_last_decode_yield,100.0)

    def test_layer_marker_requires_actual_decode_and_skips_final_layer(self):
        from .model import Transformer
        blocks=[types.SimpleNamespace(layer_id=i) for i in range(8)]
        m=types.SimpleNamespace(blocks=blocks,_prefill_forward_yielded=False,
                                _prefill_yield_decode=Mock(return_value=False))
        Transformer._prefill_layer_boundary(m,blocks[3])
        self.assertFalse(m._prefill_forward_yielded)
        m._prefill_yield_decode.return_value=True
        Transformer._prefill_layer_boundary(m,blocks[3])
        self.assertTrue(m._prefill_forward_yielded)
        Transformer._prefill_layer_boundary(m,blocks[-1])
        self.assertEqual(m._prefill_yield_decode.call_count,2)

    def test_stale_graph_or_insufficient_capacity_defers_decode(self):
        for stale in (False,True):
            e,req=self.engine()
            e._decode_active_once=Mock()
            if stale: e.rt_mtp_single._graph_cache_signature=('old',)
            else: req.pos=99
            e._yield_prefill_decode()
            e._decode_active_once.assert_not_called()
            e.rt_mtp_single._prepare_decode_cache.assert_not_called()
            self.assertEqual(e.prefill_interleave_stats['deferred_for_cache'],1)

    def test_no_handoff_without_active_slots(self):
        e,req=self.engine()
        e._active_slots={}
        e._decode_active_once=Mock()
        e._yield_prefill_decode()
        e._decode_active_once.assert_not_called()
        e._sync_prefill_devices.assert_not_called()

    def test_multiple_requests_rotate_without_touching_prefill_slot(self):
        e, first = self.engine()
        second = types.SimpleNamespace(pos=20, out_tokens=[2])
        e._active_slots[3] = second
        visited = []
        def decode(*, active_ids):
            self.assertEqual(len(active_ids), 1)
            sid = active_ids[0]
            self.assertNotEqual(sid, 0)
            visited.append(sid)
            e._active_slots[sid].out_tokens.append(9)
            if len(visited) == 4:
                e._active_slots.clear()
        e._decode_active_once = Mock(side_effect=decode)
        e._yield_prefill_decode()
        self.assertEqual(visited, [1, 3, 1, 3])
        self.assertEqual(e.prefill_interleave_stats['decode_tokens'], 4)
        self.assertEqual(e.model.shared._current_chunk_idx, 7)

    def test_prepare_reserves_all_active_request_positions(self):
        e, first = self.engine()
        e._active_slots[3] = types.SimpleNamespace(pos=200, out_tokens=[])
        e._prepare_prefill_decode(end_pos=100)
        e.rt_mtp_single._prepare_decode_cache.assert_called_once_with(205)

    def test_near_context_end_does_not_block_other_slots(self):
        e, first = self.engine()
        e.max_seq_len = 16
        first.pos = 14
        e._active_slots[3] = types.SimpleNamespace(pos=5, out_tokens=[])
        def decode(*, active_ids):
            self.assertEqual(active_ids, [3])
            e._active_slots.pop(3)
        e._decode_active_once = Mock(side_effect=decode)
        e._yield_prefill_decode()
        e._decode_active_once.assert_called_once_with(active_ids=[3])

    def test_reentrant_callback_does_not_decode_again(self):
        e,req=self.engine()
        def decode(**kwargs):
            e._yield_prefill_decode()
            e._active_slots.clear()
        e._decode_active_once=Mock(side_effect=decode)
        e._yield_prefill_decode()
        self.assertEqual(e._decode_active_once.call_count,1)


if __name__ == '__main__':
    unittest.main()
