from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pconv_unet import FinalFuse, PConv2d, PConvBlock, SpatialInpaintUNet, UpBlock, align_mask


def _load_partial_state(model: nn.Module, state, key: Optional[str] = None) -> int:
    if isinstance(state, dict) and key is not None and key in state:
        state_dict = state[key]
    else:
        state_dict = state
    model_dict = model.state_dict()
    filtered = {}
    for name, value in state_dict.items():
        clean_name = name[7:] if name.startswith("module.") else name
        if clean_name in model_dict and model_dict[clean_name].shape == value.shape:
            filtered[clean_name] = value
    model.load_state_dict(filtered, strict=False)
    return len(filtered)


class ConvINAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, groups: int = 1):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )


class PConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        self.pconv = PConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.norm = nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=True)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x, mask = self.pconv(x, mask)
        x = self.norm(x)
        x = self.act(x)
        return x, mask


class PConvResConv(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = PConv2d(channels, channels, kernel_size=3)
        self.norm = nn.InstanceNorm2d(channels, affine=True, track_running_stats=True)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = x
        y, mask = self.conv(x, mask)
        y = self.norm(y)
        y = y + residual
        y = self.act(y)
        return y, mask


class ChannelSE(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.pool(x)
        scale = self.act(self.fc1(scale))
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale


class SpatialEnhancementStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_enhancement: bool = True):
        super().__init__()
        self.use_enhancement = use_enhancement
        self.downsample = PConvNormAct(in_channels, out_channels, kernel_size=3, stride=2)
        distilled_channels = max(out_channels // 4, 1)

        self.distill1 = PConvNormAct(out_channels, distilled_channels, kernel_size=1)
        self.res1 = PConvResConv(out_channels)
        self.distill2 = PConvNormAct(out_channels, distilled_channels, kernel_size=1)
        self.res2 = PConvResConv(out_channels)
        self.distill3 = PConvNormAct(out_channels, distilled_channels, kernel_size=1)
        self.res3 = PConvResConv(out_channels)
        fusion_channels = out_channels + distilled_channels * 3
        self.local_fusion = PConvNormAct(fusion_channels, out_channels, kernel_size=1)
        self.refine = PConvNormAct(out_channels, out_channels, kernel_size=3)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x, mask = self.downsample(x, mask)
        if not self.use_enhancement:
            return x, mask

        d1, m1 = self.distill1(x, mask)
        r1, mr1 = self.res1(x, mask)
        d2, m2 = self.distill2(r1, mr1)
        r2, mr2 = self.res2(r1, mr1)
        d3, m3 = self.distill3(r2, mr2)
        r4, m4 = self.res3(r2, mr2)

        merged_mask = torch.maximum(torch.maximum(m1, m2), torch.maximum(m3, m4))
        fused, fused_mask = self.local_fusion(torch.cat([d1, d2, d3, r4], dim=1), merged_mask)
        return self.refine(fused, fused_mask)


class FrequencyEnhancementStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_enhancement: bool = True):
        super().__init__()
        self.use_enhancement = use_enhancement
        self.downsample = PConvNormAct(in_channels, out_channels, kernel_size=3, stride=2)
        self.out_norm = nn.InstanceNorm2d(out_channels, affine=True)
        self.out_act = nn.LeakyReLU(0.2, inplace=True)

        half_channels = max(out_channels // 2, 1)
        self.half_channels = half_channels
        self.fc = nn.Linear(out_channels, out_channels)
        self.vertical_band_scale = nn.Parameter(torch.ones(3, half_channels))
        self.horizontal_band_scale = nn.Parameter(torch.ones(3, half_channels))

    def _apply_band_scaling(self, fft_tensor: torch.Tensor, scales: torch.Tensor, freq_dim: int) -> torch.Tensor:
        parts = list(torch.chunk(fft_tensor, chunks=3, dim=freq_dim))
        scaled_parts = []
        for idx, part in enumerate(parts):
            view_shape = [1, part.shape[1]] + [1] * (part.ndim - 2)
            scale = scales[idx].view(*view_shape).to(dtype=part.real.dtype, device=part.device)
            scaled_parts.append(part * scale)
        return torch.cat(scaled_parts, dim=freq_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x, mask = self.downsample(x, mask)
        if not self.use_enhancement:
            return x, mask

        x_vertical = x[:, : self.half_channels]
        x_horizontal = x[:, self.half_channels : self.half_channels * 2]

        fft_vertical = torch.fft.rfft(x_vertical, dim=2, norm="ortho")
        fft_horizontal = torch.fft.rfft(x_horizontal, dim=3, norm="ortho")

        fft_vertical = self._apply_band_scaling(fft_vertical, self.vertical_band_scale, freq_dim=2)
        fft_horizontal = self._apply_band_scaling(fft_horizontal, self.horizontal_band_scale, freq_dim=3)

        fft_vertical_cat = torch.cat([fft_vertical.real, fft_vertical.imag], dim=1).permute(0, 2, 3, 1)
        fft_horizontal_cat = torch.cat([fft_horizontal.real, fft_horizontal.imag], dim=1).permute(0, 2, 3, 1)
        fft_vertical_cat = self.fc(fft_vertical_cat).permute(0, 3, 1, 2)
        fft_horizontal_cat = self.fc(fft_horizontal_cat).permute(0, 3, 1, 2)

        vertical_real = fft_vertical_cat[:, : self.half_channels]
        vertical_imag = fft_vertical_cat[:, self.half_channels : self.half_channels * 2]
        horizontal_real = fft_horizontal_cat[:, : self.half_channels]
        horizontal_imag = fft_horizontal_cat[:, self.half_channels : self.half_channels * 2]

        fft_vertical_processed = torch.complex(vertical_real, vertical_imag)
        fft_horizontal_processed = torch.complex(horizontal_real, horizontal_imag)
        vertical_out = torch.fft.irfft(fft_vertical_processed, n=x.shape[2], dim=2, norm="ortho")
        horizontal_out = torch.fft.irfft(fft_horizontal_processed, n=x.shape[3], dim=3, norm="ortho")

        enhanced = torch.cat([vertical_out, horizontal_out], dim=1)
        out = self.out_act(self.out_norm(x + enhanced))
        return out, mask


class FrequencyOnlyInpaintUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        num_levels: int = 6,
        max_channels: int = 512,
    ):
        super().__init__()
        if num_levels < 3:
            raise ValueError("num_levels must be >= 3")
        self.channel_list = self._build_channels(base_channels, num_levels, max_channels)
        channels = self.channel_list
        self.enc0 = PConvBlock(in_channels, channels[0], kernel_size=7, norm_type="instance")
        self.downs = nn.ModuleList(
            [FrequencyEnhancementStage(channels[idx - 1], channels[idx], use_enhancement=True) for idx in range(1, num_levels)]
        )
        ups = []
        current_channels = channels[-1]
        for skip_level in range(num_levels - 2, 0, -1):
            ups.append(
                UpBlock(
                    current_channels + channels[skip_level],
                    channels[skip_level],
                    kernel_size=5 if skip_level >= num_levels - 3 else 3,
                    norm_type="instance",
                )
            )
            current_channels = channels[skip_level]
        self.ups = nn.ModuleList(ups)
        self.final_fuse = FinalFuse(channels[1] + channels[0], channels[0], kernel_size=5, norm_type="instance")
        self.out_conv = nn.Conv2d(channels[0], in_channels, kernel_size=3, padding=1)
        self.out_activation = nn.Sigmoid()

    def _build_channels(self, base_channels: int, num_levels: int, max_channels: int):
        channels = []
        current = base_channels
        for _ in range(num_levels):
            channels.append(min(current, max_channels))
            current = min(current * 2, max_channels)
        return channels

    def forward(
        self,
        x: torch.Tensor,
        natural_mask: torch.Tensor,
        synthetic_mask: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,
    ):
        del state
        mask = natural_mask if synthetic_mask is None else torch.clamp(natural_mask + synthetic_mask, 0, 1)
        valid_mask = 1.0 - mask
        feat, feat_mask = self.enc0(x, valid_mask)
        skips = [feat]
        skip_masks = [feat_mask]
        current = feat
        current_mask = feat_mask
        for idx, down in enumerate(self.downs):
            current, current_mask = down(current, current_mask)
            if idx < len(self.downs) - 1:
                skips.append(current)
                skip_masks.append(current_mask)

        y = current
        y_mask = current_mask
        for idx, up in enumerate(self.ups):
            skip = skips[-(idx + 1)]
            skip_mask = align_mask(skip_masks[-(idx + 1)], skip.shape[2:])
            y, y_mask = up(y, y_mask, skip, skip_mask)
        y, y_mask = self.final_fuse(y, y_mask, skips[0], skip_masks[0])
        out = self.out_activation(self.out_conv(y))
        return out, None


class FixedGate(nn.Module):
    def forward(self, spatial_feat: torch.Tensor, freq_feat: torch.Tensor):
        return torch.ones_like(spatial_feat), torch.ones_like(freq_feat)


class SFFM(nn.Module):
    def __init__(self, channels: int, adaptive_gate: bool = True):
        super().__init__()
        self.adaptive_gate = adaptive_gate
        self.fixed_gate = FixedGate()
        self.gate_conv1 = ConvINAct(channels * 2, channels, kernel_size=3)
        self.gate_se = ChannelSE(channels)
        self.gate_conv2 = nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1, bias=True)

        self.fusion_conv1 = ConvINAct(channels * 2, channels, kernel_size=1)
        self.dw_refine = ConvINAct(channels, channels, kernel_size=3, groups=channels)
        self.fusion_conv2 = ConvINAct(channels, channels, kernel_size=1)

    def forward(
        self,
        spatial_feat: torch.Tensor,
        spatial_mask: torch.Tensor,
        freq_feat: torch.Tensor,
        freq_mask: torch.Tensor,
    ):
        if spatial_feat.shape[2:] != freq_feat.shape[2:]:
            freq_feat = F.interpolate(freq_feat, size=spatial_feat.shape[2:], mode="bilinear", align_corners=False)
            freq_mask = align_mask(freq_mask, spatial_feat.shape[2:])

        if self.adaptive_gate:
            gate_feat = self.gate_conv1(torch.cat([spatial_feat, freq_feat], dim=1))
            gate_feat = self.gate_se(gate_feat)
            gates = torch.sigmoid(self.gate_conv2(gate_feat))
            g_f2s, g_s2f = torch.chunk(gates, 2, dim=1)
        else:
            g_f2s, g_s2f = self.fixed_gate(spatial_feat, freq_feat)

        spatial_updated = spatial_feat + (freq_feat * g_f2s)
        freq_updated = freq_feat + (spatial_feat * g_s2f)
        shared_mask = torch.clamp(spatial_mask + freq_mask, 0, 1)

        fused = self.fusion_conv1(torch.cat([spatial_updated, freq_updated], dim=1))
        fused = self.dw_refine(fused)
        fused = self.fusion_conv2(fused)
        return spatial_updated, shared_mask, freq_updated, shared_mask, fused, shared_mask


@dataclass(frozen=True)
class GeneratorConfig:
    in_channels: int = 3
    base_channels: int = 64
    num_levels: int = 6
    max_channels: int = 512
    fusion_levels: Sequence[int] = (1, 2, 3, 4)


class DualBranchSFFMGenerator(nn.Module):
    def __init__(self, config: GeneratorConfig, pretrained_path: Optional[str] = None):
        super().__init__()
        if config.num_levels < 3:
            raise ValueError("num_levels must be >= 3")
        self.in_channels = config.in_channels
        self.channel_list = self._build_channels(config.base_channels, config.num_levels, config.max_channels)
        self.num_levels = config.num_levels
        default_fusion_levels = tuple(range(1, config.num_levels - 1))
        self.fusion_levels = set(config.fusion_levels or default_fusion_levels)

        channels = self.channel_list
        self.enc0 = PConvBlock(config.in_channels, channels[0], kernel_size=7, norm_type="instance")
        self.freq_enc0 = PConvBlock(config.in_channels, channels[0], kernel_size=7, norm_type="instance")
        self.spatial_stages = nn.ModuleList(
            [
                SpatialEnhancementStage(
                    channels[idx - 1],
                    channels[idx],
                    use_enhancement=True,
                )
                for idx in range(1, config.num_levels)
            ]
        )
        self.frequency_stages = nn.ModuleList(
            [
                FrequencyEnhancementStage(
                    channels[idx - 1],
                    channels[idx],
                    use_enhancement=True,
                )
                for idx in range(1, config.num_levels)
            ]
        )

        self.sffm_blocks = nn.ModuleDict()
        for level in range(1, config.num_levels - 1):
            key = f"level_{level}"
            self.sffm_blocks[key] = SFFM(channels[level], adaptive_gate=True)

        ups = []
        current_channels = channels[-1]
        for skip_level in range(config.num_levels - 2, 0, -1):
            ups.append(
                UpBlock(
                    current_channels + channels[skip_level],
                    channels[skip_level],
                    kernel_size=5 if skip_level >= config.num_levels - 3 else 3,
                    norm_type="instance",
                )
            )
            current_channels = channels[skip_level]
        self.ups = nn.ModuleList(ups)
        self.final_fuse = FinalFuse(channels[1] + channels[0], channels[0], kernel_size=5, norm_type="instance")
        self.out_conv = nn.Conv2d(channels[0], config.in_channels, kernel_size=3, padding=1)
        self.out_activation = nn.Sigmoid()

        if pretrained_path:
            state = torch.load(pretrained_path, map_location="cpu")
            matched = 0
            for key in ("generator", "model", None):
                matched = max(matched, _load_partial_state(self, state, key=key))
            print(f"Loaded {matched} compatible params from {pretrained_path}")

    def _build_channels(self, base_channels: int, num_levels: int, max_channels: int):
        channels = []
        current = base_channels
        for _ in range(num_levels):
            channels.append(min(current, max_channels))
            current = min(current * 2, max_channels)
        return channels

    def forward(
        self,
        x: torch.Tensor,
        natural_mask: torch.Tensor,
        synthetic_mask: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        del state
        mask = natural_mask if synthetic_mask is None else torch.clamp(natural_mask + synthetic_mask, 0, 1)
        valid_mask = 1.0 - mask
        spatial_feat, spatial_mask = self.enc0(x, valid_mask)
        freq_feat, freq_mask = self.freq_enc0(x, valid_mask)
        skip_feats = [spatial_feat]
        skip_masks = [spatial_mask]

        for level in range(1, self.num_levels):
            spatial_feat, spatial_mask = self.spatial_stages[level - 1](spatial_feat, spatial_mask)
            freq_feat, freq_mask = self.frequency_stages[level - 1](freq_feat, freq_mask)

            if level in self.fusion_levels and level < self.num_levels - 1:
                key = f"level_{level}"
                (
                    spatial_feat,
                    spatial_mask,
                    freq_feat,
                    freq_mask,
                    fused_skip,
                    fused_mask,
                ) = self.sffm_blocks[key](spatial_feat, spatial_mask, freq_feat, freq_mask)
                skip_feats.append(fused_skip)
                skip_masks.append(fused_mask)
            elif level < self.num_levels - 1:
                skip_feats.append(spatial_feat)
                skip_masks.append(spatial_mask)

        y = spatial_feat
        y_mask = spatial_mask
        for idx, up in enumerate(self.ups):
            skip = skip_feats[-(idx + 1)]
            skip_mask = align_mask(skip_masks[-(idx + 1)], skip.shape[2:])
            y, y_mask = up(y, y_mask, skip, skip_mask)
        y, y_mask = self.final_fuse(y, y_mask, skip_feats[0], skip_masks[0])
        out = self.out_activation(self.out_conv(y))
        return out, None


def build_generator(
    in_channels: int,
    base_channels: int,
    num_levels: int,
    max_channels: int,
    fusion_levels: Sequence[int],
    pretrained_path: Optional[str] = None,
) -> nn.Module:
    config = GeneratorConfig(
        in_channels=in_channels,
        base_channels=base_channels,
        num_levels=num_levels,
        max_channels=max_channels,
        fusion_levels=fusion_levels,
    )
    return DualBranchSFFMGenerator(config, pretrained_path=pretrained_path)
