"""Distribution, generator, and fallback checks for the standalone sampler."""

import unittest

import torch

from .sampling_candidate import nucleus_weights, sample_token_candidate


def _distribution(probs, top_p, candidate_k):
    weights, ids, fast = nucleus_weights(probs, top_p, candidate_k=candidate_k)
    out = torch.zeros_like(probs)
    if ids is None:
        out.copy_(weights)
    else:
        out.scatter_(0, ids, weights)
    return out / out.sum(), fast


class SamplingCandidateTests(unittest.TestCase):
    def test_distribution_matches_full_sort_for_varied_entropy(self):
        generator = torch.Generator().manual_seed(2387)
        for logits in (
            torch.randn(4096, generator=generator) * 5,
            torch.randn(4096, generator=generator),
            torch.zeros(4096),
            torch.tensor([8.0, 7.0, 6.0, 5.0, 0.0, -2.0]),
        ):
            probs = logits.softmax(0)
            for top_p in (0.01, 0.5, 0.95, 0.999, 1.0, 0.0):
                with self.subTest(size=probs.numel(), top_p=top_p):
                    reference, _ = _distribution(probs, top_p, candidate_k=0)
                    candidate, _ = _distribution(probs, top_p, candidate_k=32)
                    torch.testing.assert_close(candidate, reference, rtol=2e-6, atol=1e-7)

    def test_fast_path_and_fallback_conditions(self):
        sharp = torch.tensor([12.0, 10.0] + [-12.0] * 1022).softmax(0)
        uniform = torch.ones(1024) / 1024
        _, sharp_fast = _distribution(sharp, 0.95, candidate_k=32)
        _, uniform_fast = _distribution(uniform, 0.95, candidate_k=32)
        self.assertTrue(sharp_fast)
        self.assertFalse(uniform_fast)

        # The 0.4/0.4 boundary is tied; selecting one with topk would change
        # the identity distribution if its tie order differs from sort.
        tied = torch.tensor([0.4, 0.4, 0.2])
        _, tied_fast = _distribution(tied, 0.3, candidate_k=1)
        self.assertFalse(tied_fast)

    def test_generator_is_repeatable_and_greedy_does_not_advance_it(self):
        logits = torch.tensor([7.0, 6.0, 5.0, -5.0])
        a = torch.Generator().manual_seed(148)
        b = torch.Generator().manual_seed(148)
        self.assertEqual(
            [sample_token_candidate(logits, 1.0, 0.95, a, candidate_k=2) for _ in range(20)],
            [sample_token_candidate(logits, 1.0, 0.95, b, candidate_k=2) for _ in range(20)],
        )
        before = a.get_state().clone()
        self.assertEqual(sample_token_candidate(logits, 0.0, 0.95, a), 0)
        torch.testing.assert_close(a.get_state(), before)

    def test_nonfinite_logits_and_invalid_temperature_fallback(self):
        logits = torch.tensor([float("nan"), float("inf"), float("-inf"), 0.0])
        for candidate_k in (1, 256):
            self.assertEqual(
                sample_token_candidate(logits, 1.0, 0.95, torch.Generator().manual_seed(8),
                                       candidate_k=candidate_k),
                1,
            )
        self.assertEqual(sample_token_candidate(logits, float("nan"), 0.95, None), 1)
        self.assertEqual(sample_token_candidate(torch.empty(0), 0.0, 0.95, None), 0)


if __name__ == "__main__":
    unittest.main()
