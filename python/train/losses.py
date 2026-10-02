"""The supervised objective used in both original baseline training stages."""

from torch import nn
from pytorch_msssim import ssim


class HybridLoss(nn.Module):
    """MSE + alpha * (1 - SSIM), computed on normalized RGB images."""

    def __init__(self, alpha=0.05):
        super().__init__()
        if alpha < 0:
            raise ValueError("SSIM weight must be nonnegative.")
        self.alpha = alpha
        self.mse = nn.MSELoss()

    def forward(self, prediction, target):
        if prediction.shape != target.shape or prediction.ndim != 4:
            raise ValueError("Loss inputs must have matching (B, 3, H, W) shapes.")
        if min(prediction.shape[-2:]) < 11:
            raise ValueError("SSIM requires images at least 11 pixels wide and high.")
        # Fail explicitly on invalid SSIM inputs; never silently replace this term.
        score = ssim(prediction, target, data_range=1.0, size_average=True)
        return self.mse(prediction, target) + self.alpha * (1.0 - score)
