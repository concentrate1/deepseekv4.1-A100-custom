import unittest
from unittest.mock import patch

import torch

from dsv41.engram import Engram


class EngramImageMaskTest(unittest.TestCase):
    def test_image_tokens_bypass_engram_gate(self):
        layer = Engram(dim=2, hc_mult=1, layout=None, table=None,
                       wkv=torch.empty(1), q_weight=torch.ones(1, 2),
                       k_weight=torch.ones(1, 2), eps=1e-6)
        x = torch.zeros(1, 2, 1, 2)
        emb = torch.zeros(1, 2, 1)
        kv = torch.cat((torch.zeros(1, 2, 2), torch.ones(1, 2, 2)), dim=-1)
        with patch('dsv41.w8.linear_w', return_value=kv):
            result = layer.apply(x, emb, torch.tensor([[True, False]]))
        self.assertTrue(torch.equal(result[:, 1], x[:, 1]))
        self.assertTrue(torch.all(result[:, 0] > 0))


if __name__ == '__main__':
    unittest.main()
