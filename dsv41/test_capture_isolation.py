"""Cache recapture must preserve scratch and active slots, including on capture failure."""
import types
import unittest
import torch
from .decode import DecodeRuntime


class CaptureIsolationTests(unittest.TestCase):
    def test_restore_all_touched_state_on_success_and_failure(self):
        for fail in (False, True):
            with self.subTest(fail=fail):
                rt = DecodeRuntime.__new__(DecodeRuntime)
                rt.B = 5
                rt.devices = ['cpu']
                rt.tok = torch.arange(5).reshape(5, 1).clone()
                rt.pos = {'cpu': torch.arange(275000, 275005)}
                rt.seq = {'cpu': torch.tensor([1, 1, 1, 2, 2])}
                rt.pmax = {'cpu': torch.full((5,), 275004)}
                tensors = []
                def state():
                    t = torch.arange(3 * 16 * 2).reshape(3, 16, 2).float()
                    tensors.append(t)
                    return t
                compressor = types.SimpleNamespace(**{k:state() for k in ('kv_state','score_state','kv_ring','score_ring')})
                attention = types.SimpleNamespace(window_kv_cache=state(), compressor=compressor)
                shared = types.SimpleNamespace(compress_kv={(0,'cpu'):state()},index_k={(0,'cpu'):state()})
                rt.m = types.SimpleNamespace(blocks=[types.SimpleNamespace(attn=attention)],shared=shared)
                all_before = [t.clone() for t in tensors]
                inputs = [rt.tok, rt.pos['cpu'],rt.seq['cpu'],rt.pmax['cpu']]
                inputs_before = [t.clone() for t in inputs]
                try:
                    with rt._capture_in_scratch_slot():
                        self.assertEqual(rt.seq['cpu'].tolist(), [0]*5)
                        self.assertEqual(rt.pos['cpu'].tolist(), list(range(5)))
                        attention.window_kv_cache[0].fill_(-99)
                        for value in vars(compressor).values(): value[0].fill_(-99)
                        for table in (shared.compress_kv,shared.index_k):
                            for value in table.values():
                                value[0,:5].fill_(-99)
                                value[0,-1:].fill_(-99)
                        if fail: raise RuntimeError('capture failed')
                except RuntimeError:
                    self.assertTrue(fail)
                for actual, expected in zip(tensors,all_before):
                    self.assertTrue(torch.equal(actual,expected))
                for actual, expected in zip(inputs,inputs_before):
                    self.assertTrue(torch.equal(actual,expected))


if __name__ == '__main__':
    unittest.main()
