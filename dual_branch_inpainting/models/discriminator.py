from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=5, stride=2, padding=2),
            nn.InstanceNorm2d(base_channels * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=5, stride=2, padding=2),
            nn.InstanceNorm2d(base_channels * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=5, stride=2, padding=2),
            nn.InstanceNorm2d(base_channels * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 8, base_channels * 8, kernel_size=5, stride=2, padding=2),
            nn.InstanceNorm2d(base_channels * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.final_conv = nn.Conv2d(base_channels * 8, 1, kernel_size=1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, 0.0, 0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv_layers(x)
        out = self.global_pool(self.final_conv(features))
        return out.view(out.size(0), -1)


class LocalDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 64, patch_size: int = 64):
        super().__init__()
        self.patch_size = patch_size
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(base_channels * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(base_channels * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(base_channels * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.final_conv = nn.Conv2d(base_channels * 8, 1, kernel_size=1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, 0.0, 0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv_layers(x)
        out = self.global_pool(self.final_conv(features))
        return out.view(out.size(0), -1)


class GLCICDiscriminator(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        global_base_channels: int = 64,
        local_base_channels: int = 64,
        local_patch_size: int = 128,
        num_local_patches: int = 4,
        use_spectral_norm: bool = False,
    ):
        super().__init__()
        self.local_patch_size = local_patch_size
        self.num_local_patches = max(1, int(num_local_patches))
        self.global_discriminator = GlobalDiscriminator(in_channels, global_base_channels)
        self.local_discriminator = LocalDiscriminator(in_channels, local_base_channels, local_patch_size)
        if use_spectral_norm:
            self._apply_spectral_norm()

    def _apply_spectral_norm(self) -> None:
        for module in [self.global_discriminator, self.local_discriminator]:
            for submodule in module.modules():
                if isinstance(submodule, (nn.Conv2d, nn.Linear)):
                    nn.utils.spectral_norm(submodule)

    def extract_local_patches(self, images: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = images.shape
        patch_size = self.local_patch_size
        half_patch = patch_size // 2
        patches = []
        for batch_idx in range(batch):
            mask_i = masks[batch_idx, 0]
            image_i = images[batch_idx]
            hole_coords = torch.nonzero(mask_i > 0.5, as_tuple=False)
            if hole_coords.shape[0] > 0:
                num_points = hole_coords.shape[0]
                if num_points >= self.num_local_patches:
                    indices = torch.randperm(num_points, device=hole_coords.device)[: self.num_local_patches]
                else:
                    indices = torch.randint(0, num_points, (self.num_local_patches,), device=hole_coords.device)
                centers = hole_coords[indices]
            else:
                centers = torch.tensor([[height // 2, width // 2]], device=images.device, dtype=torch.long).repeat(
                    self.num_local_patches, 1
                )

            sample_patches = []
            for center in centers:
                y1 = torch.clamp(center[0] - half_patch, 0, height - 1).long()
                y2 = torch.clamp(center[0] + half_patch, 1, height).long()
                x1 = torch.clamp(center[1] - half_patch, 0, width - 1).long()
                x2 = torch.clamp(center[1] + half_patch, 1, width).long()
                patch = image_i[:, y1:y2, x1:x2]
                patch_h, patch_w = patch.shape[1], patch.shape[2]
                if patch_h < patch_size or patch_w < patch_size:
                    pad_top = (patch_size - patch_h) // 2
                    pad_bottom = patch_size - patch_h - pad_top
                    pad_left = (patch_size - patch_w) // 2
                    pad_right = patch_size - patch_w - pad_left
                    patch = F.pad(
                        patch.unsqueeze(0),
                        (pad_left, pad_right, pad_top, pad_bottom),
                        mode="reflect",
                    ).squeeze(0)
                sample_patches.append(patch[:, :patch_size, :patch_size])
            patches.append(torch.stack(sample_patches, dim=0))
        return torch.stack(patches, dim=0)

    def forward(self, images: torch.Tensor, masks: torch.Tensor, return_separate: bool = False):
        global_pred = self.global_discriminator(images)
        local_patches = self.extract_local_patches(images, masks)
        batch, num_patches, channels, patch_h, patch_w = local_patches.shape
        local_pred = self.local_discriminator(local_patches.view(batch * num_patches, channels, patch_h, patch_w))
        local_pred = local_pred.view(batch, num_patches, -1).mean(dim=1)
        if return_separate:
            return global_pred, local_pred
        return global_pred + local_pred

