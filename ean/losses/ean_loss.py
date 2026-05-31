from __future__ import annotations

import torch
import torch.nn.functional as F


def ean_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    next_latent_pred: torch.Tensor,
    next_latent_target: torch.Tensor | None = None,
    routing_weights_full: torch.Tensor | None = None,
    task: str = "classification",
    prediction_weight: float = 0.1,
    entropy_weight: float = 0.001,
) -> dict[str, torch.Tensor]:
    if task == "classification":
        task_loss = F.cross_entropy(output, target)
    elif task == "regression":
        task_loss = F.mse_loss(output, target)
    else:
        raise ValueError("task must be 'classification' or 'regression'")

    if next_latent_target is None:
        pred_loss = torch.zeros((), device=output.device)
    else:
        pred_loss = F.mse_loss(next_latent_pred, next_latent_target.detach())

    if routing_weights_full is None:
        entropy_loss = torch.zeros((), device=output.device)
    else:
        probs = routing_weights_full.clamp_min(1e-8)
        entropy = -(probs * probs.log()).sum(dim=-1).mean()
        entropy_loss = -entropy

    total = task_loss + prediction_weight * pred_loss + entropy_weight * entropy_loss
    return {
        "total": total,
        "task": task_loss.detach(),
        "prediction": pred_loss.detach(),
        "entropy_regularizer": entropy_loss.detach(),
    }
