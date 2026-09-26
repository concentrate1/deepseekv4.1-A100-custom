"""Standalone, opt-in top-p sampler candidate for small verifier batches.

The running engine does not import this module.  The fast path only applies
when a small top-k prefix contains the nucleus.  Otherwise it uses the same
full-sort rule as ``engine.sample_token``.
"""

from __future__ import annotations

import torch


def nucleus_weights(
    probs: torch.Tensor, top_p: float, *, candidate_k: int = 256
) -> tuple[torch.Tensor, torch.Tensor | None, bool]:
    """Return unnormalized weights, optional token IDs, and fast-path status.

    ``probs`` is a one-dimensional softmax result.  The chosen support uses
    ``(cumsum - probability) < top_p``, including the first token whose mass
    crosses top_p.  The full-sort path deliberately keeps zero weights, like
    the current sampler, so it can serve as a distribution oracle.
    """
    if probs.ndim != 1 or probs.numel() == 0:
        raise ValueError("expected a nonempty probability vector")
    if not 0 < top_p < 1:
        return probs, None, False

    vocab = probs.numel()
    if 0 < candidate_k < vocab:
        # One extra entry lets us detect a tie crossing the nucleus boundary.
        values, token_ids = probs.topk(candidate_k + 1, sorted=True)
        prefix = values.cumsum(0)
        if bool((prefix[candidate_k - 1] >= top_p).item()):
            keep = (prefix[:candidate_k] - values[:candidate_k]) < top_p
            keep[0] = True
            count = int(keep.sum().item())
            # topk may choose a different member of a tied boundary group.
            # Fall back so that tied tokens retain the full-sort behavior.
            if bool((values[count - 1] != values[count]).item()):
                return values[:count], token_ids[:count], True

    values, token_ids = probs.sort(descending=True)
    keep = (values.cumsum(0) - values) < top_p
    keep[0] = True
    return values * keep, token_ids, False


def sample_token_candidate(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
    gen: torch.Generator | None,
    *,
    candidate_k: int = 256,
) -> int:
    """Sample with current sanitization and fallback behavior.

    The same generator is passed to ``torch.multinomial``.  Exact token-for-
    token agreement with the full-sort sampler for a fixed seed is not
    promised: multinomial may consume randomness differently for a shortened
    vector.  The categorical distribution and repeatability are preserved.
    """
    if temperature <= 0:
        try:
            return int(logits.argmax(dim=-1).item())
        except Exception:
            return 0
    try:
        logits_f = torch.nan_to_num(logits.float(), nan=-1e4, posinf=1e4, neginf=-1e4)
        scaled = logits_f / max(float(temperature), 1e-4)
        probs = torch.nan_to_num(torch.softmax(scaled, dim=-1), nan=0.0, posinf=0.0, neginf=0.0)
        weights, token_ids, _ = nucleus_weights(probs, top_p, candidate_k=candidate_k)
        mass = weights.sum()
        if mass <= 0 or torch.isnan(mass) or torch.isinf(mass):
            return int(logits_f.argmax(dim=-1).item())
        normalized = weights / mass
        if torch.isnan(normalized).any() or torch.isinf(normalized).any() or (normalized < 0).any():
            return int(logits_f.argmax(dim=-1).item())
        selected = torch.multinomial(normalized, 1, generator=gen)
        if token_ids is not None:
            selected = token_ids.gather(-1, selected)
        return int(selected.item())
    except Exception:
        try:
            return int(logits.argmax(dim=-1).item())
        except Exception:
            return 0
