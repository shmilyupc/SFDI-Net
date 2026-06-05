from __future__ import annotations

import random
from typing import Tuple

import torch


class VerticalStripeMaskGenerator:
    def __init__(
        self,
        min_stripe: int = 30,
        max_stripe: int = 35,
        min_gap: int = 40,
        max_gap: int = 60,
        coverage_range: Tuple[float, float] = (0.18, 0.23),
        random_invert: bool = False,
        skew_range: Tuple[int, int] = (-20, 20),
    ):
        self.min_stripe = min_stripe
        self.max_stripe = max_stripe
        self.min_gap = min_gap
        self.max_gap = max_gap
        self.coverage_range = coverage_range
        self.random_invert = random_invert
        self.skew_range = skew_range

    def __call__(self, height: int, width: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        mask = torch.zeros((1, height, width), dtype=torch.float32, device=device)
        skew = random.randint(self.skew_range[0], self.skew_range[1])
        target_coverage = random.uniform(*self.coverage_range)
        covered = 0
        total = height * width
        x_base = random.randint(0, max(0, width - 1))

        while covered / total < target_coverage:
            stripe_width = random.randint(self.min_stripe, self.max_stripe)
            gap_width = random.randint(self.min_gap, self.max_gap)
            for row in range(height):
                row_offset = int(skew * row / max(height, 1))
                x0 = (x_base + row_offset) % width
                x1 = x0 + stripe_width
                if x1 <= width:
                    mask[:, row, x0:x1] = 1.0
                else:
                    mask[:, row, x0:width] = 1.0
                    mask[:, row, 0:(x1 % width)] = 1.0
            covered += height * stripe_width
            x_base = (x_base + stripe_width + gap_width) % width

        if self.random_invert and random.random() < 0.5:
            mask = 1.0 - mask
        return mask

