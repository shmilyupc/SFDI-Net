from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn

from dual_branch_inpainting.models.dual_branch import build_generator


def load_partial_state(model: nn.Module, state, key: Optional[str] = None) -> int:
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


def build_generator_model(
    in_channels: int,
    base_ch: int,
    num_levels: int,
    max_channels: int,
    pretrained_path: Optional[str],
    fusion_levels: Sequence[int],
) -> nn.Module:
    return build_generator(
        in_channels=in_channels,
        base_channels=base_ch,
        num_levels=num_levels,
        max_channels=max_channels,
        fusion_levels=fusion_levels,
        pretrained_path=pretrained_path,
    )
