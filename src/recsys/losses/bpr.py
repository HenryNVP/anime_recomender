from __future__ import annotations

import torch


def bpr_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    """Bayesian Personalized Ranking loss.

    Args:
        pos_scores: Scores for observed (positive) user-item pairs.
        neg_scores: Scores for sampled negative items (same shape as pos_scores).

    Returns:
        Scalar tensor with the mean BPR loss.
    """
    if pos_scores.shape != neg_scores.shape:
        raise ValueError(
            f"bpr_loss expects pos/neg scores with identical shapes; "
            f"got {tuple(pos_scores.shape)} vs {tuple(neg_scores.shape)}"
        )
    # softplus(x) = log(1 + exp(x)) keeps computation stable
    return torch.nn.functional.softplus(neg_scores - pos_scores).mean()
