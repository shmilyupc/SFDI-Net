from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def align_mask(mask: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    if mask.shape[2:] == size:
        return mask
    return F.interpolate(mask, size=size, mode="nearest")


class PConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = kernel_size // 2
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        nn.init.kaiming_normal_(self.weight, nonlinearity="leaky_relu")

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.pad > 0:
            x_pad = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode="reflect")
            m_pad = F.pad(mask, (self.pad, self.pad, self.pad, self.pad), mode="constant", value=0.0)
        else:
            x_pad = x
            m_pad = mask
        with torch.no_grad():
            kernel = torch.ones((1, 1, self.kernel_size, self.kernel_size), device=x.device, dtype=x.dtype)
            valid_count = F.conv2d(m_pad, kernel, stride=self.stride)
            valid_mask = valid_count > 0
        out = F.conv2d(x_pad * m_pad, self.weight, bias=None, stride=self.stride)
        norm = (self.kernel_size * self.kernel_size) / (valid_count + 1e-8)
        out = out * norm
        out = torch.where(valid_mask, out, torch.zeros_like(out))
        if self.bias is not None:
            out = torch.where(valid_mask, out + self.bias.view(1, -1, 1, 1), out)
        mask_out = torch.where(valid_mask, torch.ones_like(valid_count), torch.zeros_like(valid_count))
        return out, mask_out


class PConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, norm_type: str = "instance"):
        super().__init__()
        self.pconv1 = PConv2d(in_channels, out_channels, kernel_size=kernel_size)
        self.pconv2 = PConv2d(out_channels, out_channels, kernel_size=kernel_size)
        self.norm1 = self._make_norm(out_channels, norm_type)
        self.norm2 = self._make_norm(out_channels, norm_type)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def _make_norm(self, channels: int, norm_type: str):
        if norm_type == "instance":
            return nn.InstanceNorm2d(channels, affine=True, track_running_stats=True)
        if norm_type == "none":
            return None
        raise ValueError(f"Unsupported norm type: {norm_type}")

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x, mask = self.pconv1(x, mask)
        if self.norm1 is not None:
            x = self.norm1(x)
        x = self.act(x)
        x, mask = self.pconv2(x, mask)
        if self.norm2 is not None:
            x = self.norm2(x)
        x = self.act(x)
        return x, mask


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, norm_type: str = "instance"):
        super().__init__()
        self.block = PConvBlock(in_channels, out_channels, kernel_size=kernel_size, norm_type=norm_type)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.block(x, mask)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, norm_type: str = "instance"):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.block = PConvBlock(in_channels, out_channels, kernel_size=kernel_size, norm_type=norm_type)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        skip: torch.Tensor,
        skip_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.up(x)
        mask = align_mask(mask, x.shape[2:])
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            mask = align_mask(mask, skip.shape[2:])
        skip_mask = align_mask(skip_mask, skip.shape[2:])
        x = torch.cat([x, skip], dim=1)
        mask = torch.clamp(mask + skip_mask, 0, 1)
        return self.block(x, mask)


class FinalFuse(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5, norm_type: str = "instance"):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.block = PConvBlock(in_channels, out_channels, kernel_size=kernel_size, norm_type=norm_type)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        skip: torch.Tensor,
        skip_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.up(x)
        mask = align_mask(mask, x.shape[2:])
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            mask = align_mask(mask, skip.shape[2:])
        skip_mask = align_mask(skip_mask, skip.shape[2:])
        x = torch.cat([x, skip], dim=1)
        mask = torch.clamp(mask + skip_mask, 0, 1)
        return self.block(x, mask)


class SpatialInpaintUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        num_levels: int = 8,
        max_channels: int = 512,
        encoder_norm_type: str = "instance",
        decoder_norm_type: str = "instance",
    ):
        super().__init__()
        if num_levels < 3:
            raise ValueError("num_levels must be >= 3")
        self.channel_list = self._build_channels(base_channels, num_levels, max_channels)
        channels = self.channel_list
        self.enc0 = PConvBlock(in_channels, channels[0], kernel_size=7, norm_type=encoder_norm_type)
        self.downs = nn.ModuleList(
            [DownBlock(channels[idx - 1], channels[idx], kernel_size=5 if idx == 1 else 3, norm_type=encoder_norm_type)
             for idx in range(1, num_levels)]
        )
        ups = []
        current_channels = channels[-1]
        for skip_level in range(num_levels - 2, 0, -1):
            ups.append(
                UpBlock(
                    current_channels + channels[skip_level],
                    channels[skip_level],
                    kernel_size=5 if skip_level >= num_levels - 3 else 3,
                    norm_type=decoder_norm_type,
                )
            )
            current_channels = channels[skip_level]
        self.ups = nn.ModuleList(ups)
        self.final_fuse = FinalFuse(channels[1] + channels[0], channels[0], kernel_size=5, norm_type=decoder_norm_type)
        self.out_conv = nn.Conv2d(channels[0], in_channels, kernel_size=3, padding=1)
        self.out_activation = nn.Sigmoid()

    def _build_channels(self, base_channels: int, num_levels: int, max_channels: int) -> List[int]:
        channels = []
        current = base_channels
        for _ in range(num_levels):
            channels.append(min(current, max_channels))
            current = min(current * 2, max_channels)
        return channels

    def encode(self, x: torch.Tensor, mask: torch.Tensor):
        feat, feat_mask = self.enc0(x, 1.0 - mask)
        skips = [feat]
        skip_masks = [feat_mask]
        current = feat
        current_mask = feat_mask
        for idx, down in enumerate(self.downs):
            current = F.avg_pool2d(current, kernel_size=2, stride=2)
            current_mask = align_mask(current_mask, current.shape[2:])
            current, current_mask = down(current, current_mask)
            if idx < len(self.downs) - 1:
                skips.append(current)
                skip_masks.append(current_mask)
        return current, current_mask, skips, skip_masks

    def decode(
        self,
        bottleneck: torch.Tensor,
        bottleneck_mask: torch.Tensor,
        skips: Sequence[torch.Tensor],
        skip_masks: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        y = bottleneck
        mask = bottleneck_mask
        for idx, up in enumerate(self.ups):
            skip = skips[-(idx + 1)]
            skip_mask = skip_masks[-(idx + 1)]
            y, mask = up(y, mask, skip, skip_mask)
        y, mask = self.final_fuse(y, mask, skips[0], skip_masks[0])
        out = self.out_activation(self.out_conv(y))
        return out

    def forward(
        self,
        x: torch.Tensor,
        natural_mask: torch.Tensor,
        synthetic_mask: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,
    ):
        del state
        mask = natural_mask if synthetic_mask is None else torch.clamp(natural_mask + synthetic_mask, 0, 1)
        bottleneck, bottleneck_mask, skips, skip_masks = self.encode(x, mask)
        return self.decode(bottleneck, bottleneck_mask, skips, skip_masks), None
