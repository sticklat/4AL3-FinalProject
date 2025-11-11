import torch
import torch.nn as nn

class L1Loss(nn.Module):
    def forward(self, pred, target):
        return torch.abs(pred - target).mean()

class HuberLoss(nn.Module):
    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta

    def forward(self, pred, target):
        diff = pred - target
        abs_diff = torch.abs(diff)
        quadratic = torch.minimum(abs_diff, torch.tensor(self.delta, device=abs_diff.device))
        linear = abs_diff - quadratic
        return 0.5 * quadratic**2 + self.delta * linear
        # mean over batch
        # (PyTorch's SmoothL1Loss is similar; this is a plain Huber.)

class LogL1Loss(nn.Module):
    """Useful when errors grow with distance; compare in log space."""
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        lp = torch.log(pred.clamp_min(self.eps))
        lt = torch.log(target.clamp_min(self.eps))
        return torch.abs(lp - lt).mean()

def get_loss(name: str):
    name = (name or "").lower()
    if name in ("l1", "mae", "abs", "absolute"):
        return L1Loss()
    if name in ("huber", "smoothl1", "smooth_l1"):
        return HuberLoss(delta=1.0)
    if name in ("logl1", "log-l1", "log_mae"):
        return LogL1Loss()
    raise ValueError(f"Unknown loss '{name}'. Try: l1, huber, logl1.")