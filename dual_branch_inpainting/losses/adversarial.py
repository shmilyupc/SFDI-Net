from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GANLoss(nn.Module):
    def __init__(self, gan_type: str = "lsgan", target_real: float = 1.0, target_fake: float = 0.0):
        super().__init__()
        if gan_type not in {"lsgan", "wgan-gp"}:
            raise ValueError(f"Unsupported GAN type: {gan_type}")
        self.gan_type = gan_type
        self.target_real = target_real
        self.target_fake = target_fake

    def forward(self, prediction: torch.Tensor, is_real: bool) -> torch.Tensor:
        if self.gan_type == "lsgan":
            target = self.target_real if is_real else self.target_fake
            return F.mse_loss(prediction, torch.full_like(prediction, target))
        return -prediction.mean() if is_real else prediction.mean()

    def get_generator_loss(self, fake_prediction: torch.Tensor) -> torch.Tensor:
        if self.gan_type == "lsgan":
            return F.mse_loss(fake_prediction, torch.full_like(fake_prediction, self.target_real))
        return -fake_prediction.mean()

