"""Expert-only participants must not become dense owners or KV destinations."""
from contextlib import nullcontext
import gc
import types
import unittest
from unittest.mock import Mock, patch
import weakref

import torch
from .expert_layout import expert_shard_layout
from .engine import detached_request_error
from .ep import EPRuntime


class ExpertOnlyTests(unittest.TestCase):
    def test_layout_covers_every_expert_once(self):
        layout=expert_shard_layout([2,0,1,3],[2,0,1,3,4],[76,76,84,84,64],384)
        self.assertEqual(layout,[(2,0,76),(0,76,76),(1,152,84),(3,236,84),(4,320,64)])
        self.assertEqual(expert_shard_layout([2,0,1,3],None,[92,99,98,95],384)[-1],(3,289,95))
        with self.assertRaises(ValueError):expert_shard_layout([2,0,1,3],[2,0,1,4],[96]*4,384)
        with self.assertRaises(ValueError):expert_shard_layout([0],[0,0],[192,192],384)
        with self.assertRaises(ValueError):expert_shard_layout([0],[0,4],[384,0],384)

    def test_kv_rows_only_follow_dense_stages(self):
        rt=EPRuntime.__new__(EPRuntime)
        rt.dense_devs=[2,0,1,3];rt.dense_idx={d:i for i,d in enumerate(rt.dense_devs)}
        rt.devs=rt.dense_devs+[4]
        rt.kv_row={2:('kv','row')};rt.ik_row={2:('index','row')};rt.seq={2:'seq'}
        rt.m=types.SimpleNamespace(shared=types.SimpleNamespace(
            compress_kv={(2,d):f'kv{d}' for d in rt.dense_devs},
            index_k={(2,d):f'index{d}' for d in rt.dense_devs}))
        with patch('dsv41.ep.p2p_copy_row') as copy:
            rt._push_cache_rows(types.SimpleNamespace(layer_id=2),2)
        self.assertEqual([c.args[0] for c in copy.call_args_list],['kv0','kv1','kv3','index0','index1','index3'])

    def runtime(self):
        rt=EPRuntime.__new__(EPRuntime)
        blocks=[types.SimpleNamespace(device=2,attn=types.SimpleNamespace(is_kv_source=False,indexer=None)),
                types.SimpleNamespace(device=3,attn=types.SimpleNamespace(is_kv_source=False,indexer=None))]
        rt.m=types.SimpleNamespace(blocks=blocks,norm_w=torch.ones(2),head=torch.ones(3,2))
        rt.B=1;rt.cfg={'norm_eps':1e-6}
        rt.devs=[2,0,1,3,4];rt.idx={d:i for i,d in enumerate(rt.devs)};rt.nd=5
        rt.first_layer={2:0,3:1};rt.last_layer={2:0,3:1}
        rt.hop_h={d:torch.zeros(1,1,4,2) for d in rt.devs}
        rt._pre={d:torch.ones(1,4) for d in rt.devs}
        rt.flag_hop={d:torch.zeros(3,dtype=torch.int32) for d in rt.devs}
        rt.logits=torch.zeros(1,3);rt._wait=Mock()
        rt._owner_layer=Mock(return_value=torch.ones(1,4));rt._peer_layer=Mock()
        return rt

    def test_last_dense_layer_produces_head_even_with_later_expert_card(self):
        rt=self.runtime()
        with patch('torch.cuda.device',side_effect=lambda *_:nullcontext()), \
             patch('dsv41.ep._hc_pre',return_value=torch.ones(1,1,2)), \
             patch('dsv41.ep.rmsnorm',side_effect=lambda x,w,eps:x), \
             patch('dsv41.ep.p2p_copy',side_effect=AssertionError('unexpected pipeline hop')):
            rt.layer_section(1,3)
        self.assertTrue(torch.all(rt.logits>0))

    def test_expert_only_card_never_looks_up_dense_first_layer(self):
        rt=self.runtime()
        with patch('torch.cuda.device',side_effect=lambda *_:nullcontext()):rt.layer_section(1,4)
        rt._owner_layer.assert_not_called()
        rt._peer_layer.assert_called_once_with(rt.m.blocks[1],4)

    def test_failed_forward_tensors_are_not_retained_by_request_error(self):
        refs=[]
        def failed_forward():
            activation=torch.ones(100)
            refs.append(weakref.ref(activation))
            raise RuntimeError('own simulated OOM')
        try:failed_forward()
        except Exception as original:
            saved=detached_request_error(original)
            self.assertIsNone(saved.__traceback__)
            self.assertIsNone(saved.__context__)
            self.assertEqual(str(saved),'own simulated OOM')
            gc.collect()
            self.assertIsNone(refs[0]())


if __name__=='__main__':unittest.main()
