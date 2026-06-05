from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16


def gram_matrix(feat: torch.Tensor) -> torch.Tensor:
    batch, channels, height, width = feat.size()
    features = feat.view(batch, channels, height * width)
    gram = features @ features.transpose(1, 2)
    return gram / ((channels * height * width) + 1e-6)


class VGGFeatureExtractor(nn.Module):
    def __init__(self, layers: List[int] = [3, 8, 15, 22]):
        super().__init__()
        try:
            from torchvision.models import VGG16_Weights

            weights = getattr(VGG16_Weights, "IMAGENET1K_FEATURES", None)
            if weights is None:
                weights = getattr(VGG16_Weights, "IMAGENET1K_V1", None)
            if weights is None:
                weights = VGG16_Weights.DEFAULT
            vgg = vgg16(weights=weights).features
        except Exception:
            vgg = vgg16(pretrained=True).features

        self.slices = nn.ModuleList()
        prev = 0
        for layer in layers:
            self.slices.append(nn.Sequential(*[vgg[idx] for idx in range(prev, layer)]))
            prev = layer
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats = []
        out = x
        for part in self.slices:
            out = part(out)
            feats.append(out)
        return feats


class InpaintingLoss(nn.Module):
    def __init__(
        self,
        lambda_valid: float = 1.0,
        lambda_hole: float = 8.0,
        lambda_perc: float = 0.05,
        lambda_style: float = 30.0,
        lambda_tv: float = 2.0,
        lambda_grad: float = 0.0,
        lambda_ms_ssim: float = 0.0,
        lambda_color: float = 0.0,
    ):
        super().__init__()
        self.lambda_valid = lambda_valid
        self.lambda_hole = lambda_hole
        self.lambda_perc = lambda_perc
        self.lambda_style = lambda_style
        self.lambda_tv = lambda_tv
        self.lambda_grad = lambda_grad
        self.lambda_ms_ssim = lambda_ms_ssim
        self.lambda_color = lambda_color
        self.vgg = VGGFeatureExtractor()

    def _gaussian_kernel(self, window_size: int = 11, sigma: float = 1.5, device=None, dtype=None):
        coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
        gaussian = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
        gaussian = gaussian / (gaussian.sum() + 1e-8)
        return (gaussian[:, None] * gaussian[None, :]).contiguous()

    def _ssim_per_channel(self, x: torch.Tensor, y: torch.Tensor, window: torch.Tensor, c1: float, c2: float):
        _, channels, _, _ = x.shape
        window_size = window.shape[-1]
        weight = window.to(device=x.device, dtype=x.dtype).expand(channels, 1, window_size, window_size)
        mu_x = F.conv2d(x, weight, padding=window_size // 2, groups=channels)
        mu_y = F.conv2d(y, weight, padding=window_size // 2, groups=channels)
        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y
        sigma_x2 = F.conv2d(x * x, weight, padding=window_size // 2, groups=channels) - mu_x2
        sigma_y2 = F.conv2d(y * y, weight, padding=window_size // 2, groups=channels) - mu_y2
        sigma_xy = F.conv2d(x * y, weight, padding=window_size // 2, groups=channels) - mu_xy
        ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
            (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2) + 1e-8
        )
        cs_map = (2 * sigma_xy + c2) / (sigma_x2 + sigma_y2 + c2 + 1e-8)
        return ssim_map, cs_map

    def ms_ssim_loss(self, comp: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor, levels: int = 5):
        mask = valid_mask.expand_as(comp) if valid_mask.shape[1] == 1 and comp.shape[1] != 1 else valid_mask
        x = comp * mask
        y = target * mask
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333][:levels]
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        window = self._gaussian_kernel(11, 1.5, device=x.device, dtype=x.dtype).view(1, 1, 11, 11)
        multi_cs = []
        for idx in range(levels):
            ssim_map, cs_map = self._ssim_per_channel(x, y, window, c1, c2)
            ssim_val = ssim_map.mean(dim=(2, 3))
            cs_val = cs_map.mean(dim=(2, 3))
            if idx < levels - 1:
                multi_cs.append(cs_val)
                x = F.avg_pool2d(x, 2, 2)
                y = F.avg_pool2d(y, 2, 2)
                mask = F.avg_pool2d(mask, 2, 2)
                mask = (mask > 0.5).float()
                x = x * mask
                y = y * mask
            else:
                ssim_last = ssim_val
        ms = 1.0
        for idx in range(levels - 1):
            ms = ms * (multi_cs[idx].clamp(min=1e-6) ** weights[idx])
        ms = ms * (ssim_last.clamp(min=1e-6) ** weights[-1])
        ms = (ms.mean(dim=1).clamp(-1.0, 1.0) + 1.0) / 2.0
        return (1.0 - ms).mean()

    def tv_loss(self, x: torch.Tensor, hole_mask: torch.Tensor) -> torch.Tensor:
        dilated = F.max_pool2d(hole_mask, kernel_size=3, stride=1, padding=1)
        diff_h = (x[:, :, :, :-1] - x[:, :, :, 1:]).abs()
        diff_v = (x[:, :, :-1, :] - x[:, :, 1:, :]).abs()
        mask_h = dilated[:, :, :, :-1]
        mask_v = dilated[:, :, :-1, :]
        tv_h = (diff_h * mask_h).sum() / (mask_h.sum() + 1e-8)
        tv_v = (diff_v * mask_v).sum() / (mask_v.sum() + 1e-8)
        return tv_h + tv_v

    def color_stats_loss(self, pred: torch.Tensor, target: torch.Tensor, hole_mask: torch.Tensor) -> torch.Tensor:
        mask = hole_mask.expand_as(pred)
        denom = mask.sum(dim=(2, 3)).clamp_min(1e-8)
        pred_mean = (pred * mask).sum(dim=(2, 3)) / denom
        target_mean = (target * mask).sum(dim=(2, 3)) / denom

        pred_centered = (pred - pred_mean[:, :, None, None]) * mask
        target_centered = (target - target_mean[:, :, None, None]) * mask
        pred_std = torch.sqrt((pred_centered * pred_centered).sum(dim=(2, 3)) / denom + 1e-8)
        target_std = torch.sqrt((target_centered * target_centered).sum(dim=(2, 3)) / denom + 1e-8)

        mean_loss = (pred_mean - target_mean).abs().mean()
        std_loss = (pred_std - target_std).abs().mean()
        return mean_loss + 0.5 * std_loss

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        natural_mask: torch.Tensor,
        synthetic_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        hole = synthetic_mask.clamp(0, 1)
        natural = natural_mask.clamp(0, 1)
        valid = 1.0 - torch.clamp(natural + hole, 0, 1)
        l1_valid = (valid * (pred - target).abs()).sum() / (valid.sum() + 1e-8)
        l1_hole = (hole * (pred - target).abs()).sum() / (hole.sum() + 1e-8)
        pred_clamped = pred.clamp(0.0, 1.0)
        comp = pred_clamped * hole + target * (1.0 - hole)
        color = self.color_stats_loss(pred_clamped, target, hole)

        if comp.shape[1] == 1:
            comp_vgg = comp.repeat(1, 3, 1, 1)
            target_vgg = target.repeat(1, 3, 1, 1)
        else:
            comp_vgg = comp
            target_vgg = target

        mean = torch.tensor([0.485, 0.456, 0.406], device=pred.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=pred.device).view(1, 3, 1, 1)
        comp_n = (comp_vgg - mean) / std
        target_n = (target_vgg - mean) / std
        with torch.cuda.amp.autocast(enabled=False):
            comp_feats = self.vgg(comp_n.float())
            tgt_feats = self.vgg(target_n.float())
            perc = sum((a - b).abs().mean() for a, b in zip(comp_feats, tgt_feats))
            style = sum((gram_matrix(a) - gram_matrix(b)).abs().mean() for a, b in zip(comp_feats, tgt_feats))
        perc = torch.nan_to_num(perc, nan=0.0, posinf=1e4, neginf=0.0)
        style = torch.nan_to_num(style, nan=0.0, posinf=1e4, neginf=0.0)
        tv = self.tv_loss(pred_clamped, hole)

        dilated = F.max_pool2d(hole, kernel_size=3, stride=1, padding=1)
        grad_h = ((pred_clamped[:, :, :, 1:] - pred_clamped[:, :, :, :-1]) - (target[:, :, :, 1:] - target[:, :, :, :-1])).abs()
        grad_v = ((pred_clamped[:, :, 1:, :] - pred_clamped[:, :, :-1, :]) - (target[:, :, 1:, :] - target[:, :, :-1, :])).abs()
        mask_h = dilated[:, :, :, :-1]
        mask_v = dilated[:, :, :-1, :]
        grad = (grad_h * mask_h).sum() / (mask_h.sum() * pred_clamped.shape[1] + 1e-8)
        grad = grad + (grad_v * mask_v).sum() / (mask_v.sum() * pred_clamped.shape[1] + 1e-8)

        ms_ssim = self.ms_ssim_loss(comp, target, 1.0 - natural, levels=5)
        components = {
            "l1_valid": torch.nan_to_num(l1_valid),
            "l1_hole": torch.nan_to_num(l1_hole),
            "perceptual": torch.nan_to_num(perc),
            "style": torch.nan_to_num(style),
            "tv": torch.nan_to_num(tv),
            "grad": torch.nan_to_num(grad),
            "ms_ssim": torch.nan_to_num(ms_ssim),
            "color": torch.nan_to_num(color),
        }
        loss = (
            self.lambda_valid * components["l1_valid"]
            + self.lambda_hole * components["l1_hole"]
            + self.lambda_perc * components["perceptual"]
            + self.lambda_style * components["style"]
            + self.lambda_tv * components["tv"]
            + self.lambda_grad * components["grad"]
            + self.lambda_ms_ssim * components["ms_ssim"]
            + self.lambda_color * components["color"]
        )
        return loss, components
