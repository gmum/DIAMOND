import torch


def artifact_loss(mask: torch.Tensor, loss_cfg) -> torch.Tensor:
    mask = mask.clamp(0, 1).float()
    loss_type = loss_cfg.type

    if loss_type == "mse":
        return (mask ** 2).mean()

    if loss_type == "power":
        gamma = float(getattr(loss_cfg, "gamma", 2.0))
        return (mask ** gamma).mean()

    if loss_type == "focal":
        gamma = float(getattr(loss_cfg, "gamma", 2.0))
        alpha = float(getattr(loss_cfg, "alpha", 1.0))
        eps = 1e-6
        return (-alpha * (1 - mask) ** gamma * torch.log(1 - mask + eps)).mean()

    raise ValueError(f"Unknown loss type: {loss_type}")
