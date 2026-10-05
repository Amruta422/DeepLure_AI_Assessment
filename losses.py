"""Losses for broad-category supervised embedding learning."""
import torch
from torch.nn import functional as F


def supervised_contrastive_loss(embeddings: torch.Tensor, labels: torch.Tensor,
                                temperature: float = 0.07) -> torch.Tensor:
    """Multi-positive SupCon loss; positives mean same broad folder category only."""
    if embeddings.ndim != 2 or labels.ndim != 1 or embeddings.shape[0] != labels.shape[0]:
        raise ValueError("Expected embeddings [N,D] and labels [N]")
    n = embeddings.shape[0]
    if n < 2:
        return embeddings.sum() * 0.0
    z = F.normalize(embeddings, dim=1)
    logits = (z @ z.T) / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    eye = torch.eye(n, dtype=torch.bool, device=embeddings.device)
    positives = labels.view(-1, 1).eq(labels.view(1, -1)) & ~eye
    exp_logits = torch.exp(logits).masked_fill(eye, 0.0)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    count = positives.sum(dim=1)
    valid = count > 0
    if not torch.any(valid):
        return embeddings.sum() * 0.0
    mean_positive_log_prob = (log_prob.masked_fill(~positives, 0.0).sum(dim=1)[valid]
                              / count[valid])
    return -mean_positive_log_prob.mean()
