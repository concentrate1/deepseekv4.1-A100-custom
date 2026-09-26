"""The single-sequence indexer path must preserve multi-row score semantics."""

import unittest

import torch

from dsv41.decode import _indexer_score
from dsv41.quant import fake_quant_fp4


class IndexerScoreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41000)
        self.q = torch.randn(5, 1, 4, 128, dtype=torch.bfloat16)

    def test_single_sequence_broadcast_matches_row_gather(self):
        keys = torch.randn(1, 1024, 128, dtype=torch.bfloat16)
        seq = torch.zeros(5, dtype=torch.long)
        expected = torch.einsum("bshd,btd->bsht", self.q.float(), keys.index_select(0, seq).float())
        actual = _indexer_score(self.q, keys, seq, True, True)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_multi_sequence_keeps_row_gather(self):
        keys = torch.randn(2, 1024, 128, dtype=torch.bfloat16)
        seq = torch.tensor([1, 0, 1, 1, 0], dtype=torch.long)
        expected = torch.einsum("bshd,btd->bsht", self.q.float(), keys.index_select(0, seq).float())
        actual = _indexer_score(self.q, keys, seq, False, True)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_shared_verifier_rows_can_target_nonzero_slot(self):
        keys = torch.randn(3, 1024, 128, dtype=torch.bfloat16)
        for slot in (1, 2):
            seq = torch.full((5,), slot, dtype=torch.long)
            expected = _indexer_score(self.q, keys, seq, False, True)
            actual = _indexer_score(self.q, keys, seq, False, True, shared_sequence_rows=True)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_grouped_verifier_matches_row_gather(self):
        q = torch.cat([self.q, self.q * .5])
        keys = torch.randn(3, 1024, 128, dtype=torch.bfloat16)
        for slots in ((1, 2), (2, 0)):
            seq = torch.tensor([slots[0]] * 5 + [slots[1]] * 5)
            expected = _indexer_score(q, keys, seq, False, True)
            actual = _indexer_score(q, keys, seq, False, True, sequence_group_size=5)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_nonfused_path_keeps_direct_rows(self):
        keys = torch.randn(5, 1024, 128, dtype=torch.bfloat16)
        seq = torch.tensor([4, 3, 2, 1, 0], dtype=torch.long)
        expected = torch.einsum("bshd,btd->bsht", self.q.float(), keys.float())
        actual = _indexer_score(self.q, keys, seq, False, False)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_grouped_quantized_gpu_scores(self):
        q = fake_quant_fp4(torch.randn(10, 1, 32, 128, device="cuda:0", dtype=torch.bfloat16), 32)
        keys = fake_quant_fp4(torch.randn(3, 8192, 128, device="cuda:0", dtype=torch.bfloat16), 32)
        seq = torch.tensor([1] * 5 + [2] * 5, device="cuda:0")
        expected = _indexer_score(q, keys, seq, False, True)
        actual = _indexer_score(q, keys, seq, False, True, sequence_group_size=5)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_quantized_gpu_scores_and_topk(self):
        device = "cuda:0"
        q = fake_quant_fp4(torch.randn(5, 1, 32, 128, device=device, dtype=torch.bfloat16), 32)
        keys = fake_quant_fp4(torch.randn(1, 8192, 128, device=device, dtype=torch.bfloat16), 32)
        seq = torch.zeros(5, device=device, dtype=torch.long)
        expected = _indexer_score(q, keys, seq, False, True)
        actual = _indexer_score(q, keys, seq, True, True)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
        a = expected[..., :1024].topk(512, dim=-1).indices
        b = actual[..., :1024].topk(512, dim=-1).indices
        self.assertTrue(torch.equal(a.sort(dim=-1).values, b.sort(dim=-1).values))


if __name__ == "__main__":
    unittest.main()
