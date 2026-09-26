"""CPU checks for DSpark verifier sampling and loop safety."""
import unittest

import torch

from .engine import GenParams, mtp_loop_hit, sample_mtp_verified
from .serve import _params


class MTPSamplingTests(unittest.TestCase):
    def test_api_uses_official_temperature_unless_request_overrides_it(self):
        self.assertEqual(_params({}).temperature, 1.0)
        self.assertEqual(_params({}).top_p, 0.95)
        self.assertEqual(_params({"temperature": 0.3}).temperature, 0.3)

    def test_default_penalties_are_neutral(self):
        params = _params({})
        self.assertEqual((params.repetition_penalty, params.presence_penalty,
                          params.frequency_penalty, params.progressive_penalty),
                         (1.0, 0.0, 0.0, 0.0))
        self.assertFalse(params.ban_cycles)
        self.assertFalse(params.loop_detect)
        self.assertEqual(_params({"repetition_penalty": 1.05}).repetition_penalty, 1.05)

    def test_repeated_draft_is_rejected_by_repetition_penalty(self):
        params = GenParams(temperature=0, top_p=1, repetition_penalty=2,
                           frequency_penalty=0, progressive_penalty=0,
                           ban_cycles=False)
        logits = torch.tensor([[0.0, 5.0, 4.0, 0.0],
                               [0.0, 0.0, 0.0, 6.0]])
        accepted, bonus = sample_mtp_verified(logits, [1], [1], params, None)
        self.assertEqual((accepted, bonus), (0, 2))

    def test_default_penalties_redirect_a_frequent_token(self):
        params = GenParams(temperature=0, top_p=0.95,
                           repetition_penalty=1.05, frequency_penalty=0.02,
                           penalty_window=2048, progressive_penalty=0,
                           ban_cycles=True)
        logits = torch.tensor([[0.0, 5.0, 4.9], [0.0, 0.0, 5.0]])
        accepted, bonus = sample_mtp_verified(logits, [1], [1] * 20, params, None)
        self.assertEqual((accepted, bonus), (0, 2))

    def test_bonus_uses_accepted_draft_in_penalty_history(self):
        params = GenParams(temperature=0, top_p=1, repetition_penalty=2,
                           frequency_penalty=0, progressive_penalty=0,
                           ban_cycles=False)
        logits = torch.tensor([[0.0, 0.0, 5.0, 4.0],
                               [0.0, 0.0, 5.0, 4.0]])
        accepted, bonus = sample_mtp_verified(logits, [2], [1], params, None)
        self.assertEqual((accepted, bonus), (1, 3))

    def test_exact_repeated_output_stops_with_loop_detection_enabled(self):
        tokens = list(range(48)) * 2
        self.assertIsNotNone(mtp_loop_hit(tokens, GenParams(loop_detect=True, min_loop_match=48)))
        self.assertIsNone(mtp_loop_hit(tokens, GenParams(loop_detect=False)))


if __name__ == "__main__":
    unittest.main()
