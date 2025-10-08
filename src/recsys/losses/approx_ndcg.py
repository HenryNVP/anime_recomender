from __future__ import annotations

import torch


def _approx_ranks(scores: torch.Tensor, mask: torch.Tensor | None, temperature: float) -> torch.Tensor:
    """Approximate item ranks via pairwise sigmoid comparisons (from allRank)."""
    diff = (scores.unsqueeze(-1) - scores.unsqueeze(-2)) / max(temperature, 1e-6)
    pairwise = torch.sigmoid(-diff)
    if mask is not None:
        mask = mask.float()
        pairwise = pairwise * mask.unsqueeze(-1) * mask.unsqueeze(-2)
        diag = mask
    else:
        diag = torch.ones_like(scores)
    ranks = pairwise.sum(dim=-1) + 0.5 * diag
    return ranks


def approx_ndcg_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    temperature: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Approximate NDCG loss (1 - mean NDCG) using smooth rank estimation.

    Args:
        scores: Predicted relevance scores, shape (batch, list_len).
        labels: Ground-truth relevance (non-negative), same shape as scores.
        mask: Optional binary mask marking valid items (same shape).
        temperature: Softmax temperature controlling approximation sharpness.
        eps: Numerical stability constant.
    """
    if scores.ndim != 2 or labels.ndim != 2:
        raise ValueError("scores and labels must be 2-D tensors [batch, list_len]")
    if scores.shape != labels.shape:
        raise ValueError("scores and labels must have the same shape")
    if mask is not None and mask.shape != scores.shape:
        raise ValueError("mask must be the same shape as scores")

    mask_f = mask.float() if mask is not None else torch.ones_like(labels, dtype=torch.float32)

    approx_ranks = _approx_ranks(scores, mask_f, temperature)
    gains = torch.pow(2.0, labels) - 1.0
    discounts = 1.0 / torch.log2(approx_ranks + 1.0)

    dcg = (gains * discounts * mask_f).sum(dim=-1)

    ideal_labels, _ = torch.sort(labels, dim=-1, descending=True)
    ideal_discounts = 1.0 / torch.log2(torch.arange(1, labels.size(-1) + 1, device=labels.device).float() + 1.0)
    ideal_dcg = ((torch.pow(2.0, ideal_labels) - 1.0) * ideal_discounts).sum(dim=-1)
    ideal_dcg = torch.clamp(ideal_dcg, min=eps)

    ndcg = dcg / ideal_dcg
    loss = 1.0 - ndcg
    return loss.mean()
