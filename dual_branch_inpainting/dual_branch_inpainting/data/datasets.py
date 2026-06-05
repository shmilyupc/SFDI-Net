from __future__ import annotations

import os
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


SUPPORTED_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def list_image_mask_pairs(images_dir: str, masks_dir: str) -> List[Tuple[str, str]]:
    image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(".png")])
    pairs = []
    for img in image_files:
        mask_name = os.path.splitext(img)[0] + "_mask.png"
        img_path = os.path.join(images_dir, img)
        mask_path = os.path.join(masks_dir, mask_name)
        if os.path.exists(mask_path):
            pairs.append((img_path, mask_path))
    return pairs


def scale_width(image: np.ndarray, target_width: int) -> np.ndarray:
    height, width = image.shape[:2]
    if width <= target_width:
        return image
    return cv2.resize(image, (target_width, height), interpolation=cv2.INTER_AREA)


def pad_to_width(image: np.ndarray, final_width: int, is_mask: bool) -> np.ndarray:
    del is_mask
    height, width = image.shape[:2]
    if width >= final_width:
        return image[:, :final_width]
    pad_left = (final_width - width) // 2
    pad_right = final_width - width - pad_left
    return cv2.copyMakeBorder(image, 0, 0, pad_left, pad_right, cv2.BORDER_WRAP)


def vertical_crops(image: np.ndarray, crop_height: int, overlap: int) -> List[Tuple[int, int]]:
    height = image.shape[0]
    if height <= crop_height:
        return [(0, height)]

    stride = max(1, crop_height - overlap)
    indices: List[Tuple[int, int]] = []
    start = 0
    while start < height:
        end = min(start + crop_height, height)
        indices.append((start, end))
        if end >= height:
            break
        start += stride

    if indices and indices[-1][1] < height:
        last_start = max(0, height - crop_height)
        indices[-1] = (last_start, height)
    return indices


class ImageMaskPairDataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        target_width: int = 512,
        crop_height: int = 256,
        final_width: int = 512,
        overlap: int = 64,
        enable_cache: bool = True,
        use_grayscale: bool = False,
    ):
        self.pairs = list_image_mask_pairs(images_dir, masks_dir)
        if not self.pairs:
            raise ValueError(f"No image/mask pairs found in {images_dir} and {masks_dir}")

        self.target_width = target_width
        self.crop_height = crop_height
        self.final_width = final_width
        self.overlap = overlap
        self.enable_cache = enable_cache
        self.use_grayscale = use_grayscale
        self.index_map: List[Tuple[int, int, int]] = []
        self.cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self._precompute_indices()

    def _precompute_indices(self) -> None:
        for pair_idx, (img_path, mask_path) in enumerate(self.pairs):
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if image is None or mask is None:
                continue
            image = scale_width(image, self.target_width)
            mask = scale_width(mask, self.target_width)
            image = pad_to_width(image, self.final_width, is_mask=False)
            mask = pad_to_width(mask, self.final_width, is_mask=True)
            for y0, y1 in vertical_crops(image, self.crop_height, self.overlap):
                self.index_map.append((pair_idx, y0, y1))

    def __len__(self) -> int:
        return len(self.index_map)

    def _load_pair(self, pair_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.enable_cache and pair_idx in self.cache:
            return self.cache[pair_idx]

        image_path, mask_path = self.pairs[pair_idx]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise ValueError(f"Unable to read pair: {image_path}, {mask_path}")

        image = scale_width(image, self.target_width)
        mask = scale_width(mask, self.target_width)
        image = pad_to_width(image, self.final_width, is_mask=False)
        mask = pad_to_width(mask, self.final_width, is_mask=True)
        if self.enable_cache:
            self.cache[pair_idx] = (image, mask)
        return image, mask

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        pair_idx, y0, y1 = self.index_map[index]
        image, mask = self._load_pair(pair_idx)
        image_crop = image[y0:y1]
        mask_crop = mask[y0:y1]
        if image_crop.shape[0] != self.crop_height:
            pad_bottom = self.crop_height - image_crop.shape[0]
            image_crop = cv2.copyMakeBorder(image_crop, 0, pad_bottom, 0, 0, cv2.BORDER_REFLECT_101)
            mask_crop = cv2.copyMakeBorder(mask_crop, 0, pad_bottom, 0, 0, cv2.BORDER_REFLECT_101)

        if self.use_grayscale:
            image_gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
            image_t = torch.from_numpy(image_gray).unsqueeze(0).float() / 255.0
        else:
            image_rgb = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
            image_t = torch.from_numpy(image_rgb.transpose(2, 0, 1)).float() / 255.0
        mask_t = torch.from_numpy(mask_crop.astype(np.float32) / 255.0).unsqueeze(0)

        return {
            "image": image_t,
            "natural_mask": mask_t,
            "sequence_id": torch.tensor(pair_idx, dtype=torch.long),
            "is_new_sequence": torch.tensor(1.0 if y0 == 0 else 0.0, dtype=torch.float32),
        }
