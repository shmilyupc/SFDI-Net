from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dual_branch_inpainting.experiments import DEFAULT_FSJ_LEVELS, MAIN_EXPERIMENT_KEY, MAIN_EXPERIMENT_TITLE
from dual_branch_inpainting.factory import build_generator_model, load_partial_state
from dual_branch_inpainting.data.datasets import ImageMaskPairDataset, pad_to_width, scale_width, vertical_crops
from dual_branch_inpainting.models.discriminator import GLCICDiscriminator
from dual_branch_inpainting.losses.adversarial import GANLoss
from dual_branch_inpainting.losses.inpainting import InpaintingLoss
from dual_branch_inpainting.data.masks import VerticalStripeMaskGenerator


DATA_ROOT = REPO_ROOT / "data"
OUTPUTS_ROOT = REPO_ROOT / "outputs"

DEFAULT_PRETRAINED_UNET = ""
DEFAULT_IMAGES_DIR = str(DATA_ROOT / "train_images_with_gaps")
DEFAULT_MASKS_DIR = str(DATA_ROOT / "train_masks")
DEFAULT_TEST_IMAGES_DIR = str(DATA_ROOT / "test_images")
DEFAULT_TEST_MASKS_DIR = str(DATA_ROOT / "test_masks")


def parse_int_list(text: str) -> List[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def default_output_root() -> Path:
    return OUTPUTS_ROOT / MAIN_EXPERIMENT_KEY


def resolve_device(device_index: int) -> torch.device:
    if torch.cuda.is_available():
        torch.cuda.set_device(device_index)
        return torch.device(f"cuda:{device_index}")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_requires_grad(module: torch.nn.Module, enabled: bool) -> None:
    for param in module.parameters():
        param.requires_grad = enabled


def build_dataset(args) -> torch.utils.data.Dataset:
    return ImageMaskPairDataset(
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        target_width=args.target_width,
        crop_height=args.crop_height,
        final_width=args.final_width,
        overlap=args.overlap,
        use_grayscale=False,
        enable_cache=True,
    )


def build_dataloader(dataset, batch_size: int, num_workers: int, prefetch_factor: int) -> DataLoader:
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": True,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
        kwargs["persistent_workers"] = True
    return DataLoader(**kwargs)


def build_discriminator(in_channels: int, device: torch.device) -> GLCICDiscriminator:
    return GLCICDiscriminator(
        in_channels=in_channels,
        global_base_channels=64,
        local_base_channels=64,
        local_patch_size=64,
        num_local_patches=6,
        use_spectral_norm=False,
    ).to(device)


def build_mask_generator(args) -> VerticalStripeMaskGenerator:
    return VerticalStripeMaskGenerator(
        min_stripe=args.syn_min_stripe,
        max_stripe=args.syn_max_stripe,
        min_gap=args.syn_min_gap,
        max_gap=args.syn_max_gap,
        coverage_range=(args.syn_coverage_min, args.syn_coverage_max),
        random_invert=False,
        skew_range=(args.syn_skew_min, args.syn_skew_max),
    )


def select_state_dict_key(checkpoint: dict) -> Optional[str]:
    if "generator" in checkpoint:
        return "generator"
    if "model" in checkpoint:
        return "model"
    return None


def load_generator_checkpoint(generator: torch.nn.Module, checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    key = select_state_dict_key(checkpoint) if isinstance(checkpoint, dict) else None
    matched = load_partial_state(generator, checkpoint, key=key)
    if matched == 0:
        raise RuntimeError(f"No compatible generator weights found in {checkpoint_path}")
    return checkpoint


def checkpoint_summary(generator: torch.nn.Module) -> Tuple[int, int]:
    total_params = sum(param.numel() for param in generator.parameters())
    trainable_params = sum(param.numel() for param in generator.parameters() if param.requires_grad)
    return total_params, trainable_params


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def create_live_plotter(output_root: Path):
    metric_specs = [
        ("g_loss", "Generator Total"),
        ("d_loss", "Discriminator Total"),
        ("recon", "Reconstruction"),
        ("gan", "GAN"),
        ("l1_valid", "L1 Valid"),
        ("l1_hole", "L1 Hole"),
        ("perceptual", "Perceptual"),
        ("style", "Style"),
        ("tv", "TV"),
        ("grad", "Gradient"),
        ("ms_ssim", "MS-SSIM"),
        ("color", "Hole Color"),
        ("d_real", "D Real"),
        ("d_fake", "D Fake"),
    ]
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    display_handle = None
    update_display_fn = None
    try:
        from IPython.display import display

        fig = plt.figure(figsize=(18, 16))
        display_handle = display(fig, display_id=True)
        update_display_fn = getattr(display_handle, "update", None)
    except Exception:
        fig = plt.figure(figsize=(18, 16))

    def _update(history) -> None:
        if not history:
            return
        available_metrics = [(key, title) for key, title in metric_specs if key in history[0]]
        if not available_metrics:
            return
        epochs = [item["epoch"] for item in history]
        ncols = 3
        nrows = (len(available_metrics) + ncols - 1) // ncols
        fig.clf()
        axes = fig.subplots(nrows, ncols)
        axes = np.array(axes).reshape(-1)
        for ax, (key, title) in zip(axes, available_metrics):
            values = [item.get(key, 0.0) for item in history]
            ax.plot(epochs, values, marker="o", linewidth=1.5)
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Metric")
            ax.grid(True, alpha=0.3)
        for ax in axes[len(available_metrics) :]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_root / "loss_curves_live.png", dpi=150, bbox_inches="tight")
        if update_display_fn is not None:
            update_display_fn(fig)
        else:
            fig.canvas.draw_idle()

    return _update


def build_common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--base-ch", type=int, default=64)
    parser.add_argument("--num-levels", type=int, default=6)
    parser.add_argument("--max-channels", type=int, default=512)
    parser.add_argument("--fsj-levels", type=str, default=",".join(str(v) for v in DEFAULT_FSJ_LEVELS))
    return parser


def build_train_parser() -> argparse.ArgumentParser:
    parser = build_common_parser(f"Train {MAIN_EXPERIMENT_TITLE}")
    parser.add_argument("--pretrained-unet-path", type=str, default=DEFAULT_PRETRAINED_UNET)
    parser.add_argument("--images-dir", type=str, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--masks-dir", type=str, default=DEFAULT_MASKS_DIR)
    parser.add_argument("--output-root", type=str, default=str(default_output_root()))
    parser.add_argument("--target-width", type=int, default=512)
    parser.add_argument("--crop-height", type=int, default=256)
    parser.add_argument("--final-width", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-epochs", type=int, default=500)
    parser.add_argument("--g-lr", type=float, default=2e-6)
    parser.add_argument("--d-lr", type=float, default=1e-6)
    parser.add_argument("--lambda-l1", type=float, default=1.5)
    parser.add_argument("--lambda-perc", type=float, default=1.5)
    parser.add_argument("--lambda-style", type=float, default=60.0)
    parser.add_argument("--lambda-tv", type=float, default=0.5)
    parser.add_argument("--lambda-grad", type=float, default=12.0)
    parser.add_argument("--lambda-ms-ssim", type=float, default=25.0)
    parser.add_argument("--lambda-color", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--lambda-gan", type=float, default=0.8)
    parser.add_argument("--w-global", type=float, default=1.0)
    parser.add_argument("--w-local", type=float, default=2.0)
    parser.add_argument("--gan-type", choices=("lsgan", "wgan-gp"), default="lsgan")
    parser.add_argument("--d-steps", type=int, default=1)
    parser.add_argument("--syn-min-stripe", type=int, default=30)
    parser.add_argument("--syn-max-stripe", type=int, default=35)
    parser.add_argument("--syn-min-gap", type=int, default=40)
    parser.add_argument("--syn-max-gap", type=int, default=60)
    parser.add_argument("--syn-coverage-min", type=float, default=0.18)
    parser.add_argument("--syn-coverage-max", type=float, default=0.23)
    parser.add_argument("--syn-skew-min", type=int, default=-20)
    parser.add_argument("--syn-skew-max", type=int, default=20)
    parser.add_argument("--no-resume", action="store_true")
    return parser


def build_infer_parser() -> argparse.ArgumentParser:
    parser = build_common_parser(f"Infer {MAIN_EXPERIMENT_TITLE}")
    parser.add_argument("--checkpoint-path", type=str, default=str(default_output_root() / "checkpoints" / "latest.pt"))
    parser.add_argument("--test-images-dir", type=str, default=DEFAULT_TEST_IMAGES_DIR)
    parser.add_argument("--test-masks-dir", type=str, default=DEFAULT_TEST_MASKS_DIR)
    parser.add_argument("--infer-dir", type=str, default=str(default_output_root() / "infer"))
    parser.add_argument("--target-width", type=int, default=512)
    parser.add_argument("--crop-height", type=int, default=256)
    parser.add_argument("--final-width", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=16)
    parser.add_argument("--output-suffix", type=str, default="sfdi_net")
    return parser


def make_generator(args, pretrained_path: Optional[str], device: torch.device):
    return build_generator_model(
        in_channels=args.in_channels,
        base_ch=args.base_ch,
        num_levels=args.num_levels,
        max_channels=args.max_channels,
        pretrained_path=pretrained_path,
        fusion_levels=parse_int_list(args.fsj_levels),
    ).to(device)


def parse_args(parser: argparse.ArgumentParser, argv: Optional[Sequence[str]] = None):
    return parser.parse_args(list(argv) if argv is not None else None)


def train_main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(build_train_parser(), argv)
    output_root = Path(args.output_root)
    checkpoint_dir = output_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest_checkpoint = checkpoint_dir / "latest.pt"
    resume_checkpoint = latest_checkpoint

    set_seed(args.seed)
    cudnn.benchmark = True
    device = resolve_device(args.device_index)
    use_gan = True
    fusion_levels = parse_int_list(args.fsj_levels)
    resume_training = (not args.no_resume) and resume_checkpoint.exists()
    checkpoint = None
    start_epoch = 0
    loss_history = []
    best_recon = float("inf")
    print("=" * 60)
    print(MAIN_EXPERIMENT_TITLE)
    print("=" * 60)
    print(f"Use GAN: {use_gan}")
    print(f"Device: {device}")
    print(f"Fusion levels: {fusion_levels}")
    print(f"Output root: {output_root}")
    print("=" * 60)

    if resume_training:
        generator = make_generator(args, pretrained_path=None, device=device)
        checkpoint = load_generator_checkpoint(generator, resume_checkpoint, device)
        start_epoch = int(checkpoint.get("epoch", 0)) + 1 if isinstance(checkpoint, dict) else 0
        loss_history = checkpoint.get("loss_history", []) if isinstance(checkpoint, dict) else []
        best_recon = float(checkpoint.get("best_recon", float("inf"))) if isinstance(checkpoint, dict) else float("inf")
        print(f"Resuming from {resume_checkpoint}")
        print(f"Start epoch: {start_epoch}")
    else:
        pretrained_path = args.pretrained_unet_path.strip()
        generator = make_generator(
            args,
            pretrained_path=pretrained_path or None,
            device=device,
        )
        if pretrained_path:
            print(f"Initializing from pretrained UNet: {pretrained_path}")
        else:
            print("Initializing from scratch")

    discriminator = build_discriminator(args.in_channels, device) if use_gan else None
    if use_gan and resume_training and isinstance(checkpoint, dict) and "discriminator" in checkpoint:
        matched = load_partial_state(discriminator, checkpoint, key="discriminator")
        print(f"Loaded discriminator params: {matched}")

    total_params, trainable_params = checkpoint_summary(generator)
    print(f"Generator total params: {total_params:,}")
    print(f"Generator trainable params: {trainable_params:,}")

    dataset = build_dataset(args)
    dataloader = build_dataloader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )
    print(f"Dataset size: {len(dataset)}")

    recon_criterion = InpaintingLoss(
        lambda_valid=1.0,
        lambda_hole=args.lambda_l1,
        lambda_perc=args.lambda_perc,
        lambda_style=args.lambda_style,
        lambda_tv=args.lambda_tv,
        lambda_grad=args.lambda_grad,
        lambda_ms_ssim=args.lambda_ms_ssim,
        lambda_color=args.lambda_color,
    ).to(device)

    gan_criterion = GANLoss(gan_type=args.gan_type) if use_gan else None
    opt_g = Adam(generator.parameters(), lr=args.g_lr, betas=(0.5, 0.999))
    scheduler_g = torch.optim.lr_scheduler.StepLR(opt_g, step_size=2, gamma=0.8)
    opt_d = None
    scheduler_d = None
    if use_gan and discriminator is not None:
        opt_d = Adam(discriminator.parameters(), lr=args.d_lr, betas=(0.5, 0.999))
        scheduler_d = torch.optim.lr_scheduler.StepLR(opt_d, step_size=5, gamma=0.5)

    if resume_training and isinstance(checkpoint, dict):
        try:
            if "opt_g" in checkpoint:
                opt_g.load_state_dict(checkpoint["opt_g"])
            if "scheduler_g" in checkpoint:
                scheduler_g.load_state_dict(checkpoint["scheduler_g"])
        except (KeyError, ValueError) as exc:
            print(f"Skip generator optimizer restore: {type(exc).__name__}")
        if use_gan and opt_d is not None and scheduler_d is not None:
            try:
                if "opt_d" in checkpoint:
                    opt_d.load_state_dict(checkpoint["opt_d"])
                if "scheduler_d" in checkpoint:
                    scheduler_d.load_state_dict(checkpoint["scheduler_d"])
            except (KeyError, ValueError) as exc:
                print(f"Skip discriminator optimizer restore: {type(exc).__name__}")

    mask_generator = build_mask_generator(args)
    print("=" * 60)
    print(f"Start training: epoch {start_epoch + 1} -> {args.num_epochs}")
    print("=" * 60)

    total_epochs = max(args.num_epochs - start_epoch, 0)
    total_steps = total_epochs * len(dataloader)
    train_pbar = tqdm(total=total_steps, desc="Training", dynamic_ncols=True)
    live_plotter = create_live_plotter(output_root)

    for epoch in range(start_epoch, args.num_epochs):
        generator.train()
        if use_gan and discriminator is not None:
            discriminator.train()

        epoch_g_losses = []
        epoch_d_losses = []
        epoch_recon_losses = []
        last_g_items = {"GAN": 0.0, "l1_valid": 0.0, "l1_hole": 0.0, "perc": 0.0, "style": 0.0, "tv": 0.0, "grad": 0.0, "ms": 0.0, "color": 0.0}
        last_d_items = {"D_real": 0.0, "D_fake": 0.0}
        epoch_g_component_sums = {"GAN": 0.0, "l1_valid": 0.0, "l1_hole": 0.0, "perc": 0.0, "style": 0.0, "tv": 0.0, "grad": 0.0, "ms": 0.0, "color": 0.0}
        epoch_g_raw_component_sums = {"l1_valid": 0.0, "l1_hole": 0.0, "perc": 0.0, "style": 0.0, "tv": 0.0, "grad": 0.0, "ms": 0.0, "color": 0.0}
        epoch_d_component_sums = {"D_real": 0.0, "D_fake": 0.0}
        epoch_g_component_count = 0
        epoch_d_component_count = 0

        for batch_idx, batch in enumerate(dataloader):
            img = batch["image"].to(device, non_blocking=True)
            natural_mask = batch["natural_mask"].to(device, non_blocking=True)
            batch_size, _, height, width = img.shape
            synthetic_mask = torch.stack([mask_generator(height, width, device=device) for _ in range(batch_size)], dim=0)
            synthetic_mask = synthetic_mask * (1.0 - natural_mask)
            disc_mask = synthetic_mask.clamp(0, 1)
            disc_valid = 1.0 - natural_mask
            hole = synthetic_mask.clamp(0, 1)
            hole_c = hole.expand(-1, img.shape[1], -1, -1)

            if use_gan and discriminator is not None and opt_d is not None and gan_criterion is not None:
                set_requires_grad(generator, False)
                set_requires_grad(discriminator, True)
                real_global, real_local = discriminator(img * disc_valid, disc_mask, return_separate=True)
                d_real_loss = args.w_global * gan_criterion(real_global, True) + args.w_local * gan_criterion(real_local, True)
                with torch.no_grad():
                    fake_images, _ = generator(img, natural_mask=natural_mask, synthetic_mask=synthetic_mask, state=None)
                fake_for_d = fake_images.clamp(0.0, 1.0)
                comp_fake = fake_for_d * hole_c + img * (1.0 - hole_c)
                fake_global, fake_local = discriminator(comp_fake * disc_valid, disc_mask, return_separate=True)
                d_fake_loss = args.w_global * gan_criterion(fake_global, False) + args.w_local * gan_criterion(fake_local, False)
                d_loss = 0.5 * (d_real_loss + d_fake_loss)
                opt_d.zero_grad()
                d_loss.backward()
                opt_d.step()
                epoch_d_losses.append(d_loss.item())
                last_d_items["D_real"] = d_real_loss.item()
                last_d_items["D_fake"] = d_fake_loss.item()
                epoch_d_component_sums["D_real"] += d_real_loss.item()
                epoch_d_component_sums["D_fake"] += d_fake_loss.item()
                epoch_d_component_count += 1

            if (batch_idx + 1) % args.d_steps == 0:
                set_requires_grad(generator, True)
                if use_gan and discriminator is not None:
                    set_requires_grad(discriminator, False)
                fake_images, _ = generator(img, natural_mask=natural_mask, synthetic_mask=synthetic_mask, state=None)
                g_recon_loss, recon_comps = recon_criterion(fake_images, img.clamp(0, 1), natural_mask, synthetic_mask)
                if use_gan and discriminator is not None and gan_criterion is not None:
                    fake_for_d = fake_images.clamp(0.0, 1.0)
                    comp_fake = fake_for_d * hole_c + img * (1.0 - hole_c)
                    fake_global, fake_local = discriminator(comp_fake * disc_valid, disc_mask, return_separate=True)
                    g_gan_loss = args.w_global * gan_criterion.get_generator_loss(fake_global) + args.w_local * gan_criterion.get_generator_loss(fake_local)
                    g_loss = args.lambda_gan * g_gan_loss + g_recon_loss
                    last_g_items["GAN"] = (args.lambda_gan * g_gan_loss).item()
                else:
                    g_loss = g_recon_loss
                    last_g_items["GAN"] = 0.0

                opt_g.zero_grad()
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
                opt_g.step()
                epoch_g_losses.append(g_loss.item())
                epoch_recon_losses.append(g_recon_loss.item())
                raw_l1_valid = recon_comps["l1_valid"].item()
                raw_l1_hole = recon_comps["l1_hole"].item()
                raw_perc = recon_comps["perceptual"].item()
                raw_style = recon_comps["style"].item()
                raw_tv = recon_comps["tv"].item()
                raw_grad = recon_comps["grad"].item()
                raw_ms = recon_comps["ms_ssim"].item()
                raw_color = recon_comps["color"].item()

                last_g_items["l1_valid"] = 1.0 * raw_l1_valid
                last_g_items["l1_hole"] = args.lambda_l1 * raw_l1_hole
                last_g_items["perc"] = args.lambda_perc * raw_perc
                last_g_items["style"] = args.lambda_style * raw_style
                last_g_items["tv"] = args.lambda_tv * raw_tv
                last_g_items["grad"] = args.lambda_grad * raw_grad
                last_g_items["ms"] = args.lambda_ms_ssim * raw_ms
                last_g_items["color"] = args.lambda_color * raw_color
                epoch_g_component_sums["GAN"] += last_g_items["GAN"]
                epoch_g_component_sums["l1_valid"] += last_g_items["l1_valid"]
                epoch_g_component_sums["l1_hole"] += last_g_items["l1_hole"]
                epoch_g_component_sums["perc"] += last_g_items["perc"]
                epoch_g_component_sums["style"] += last_g_items["style"]
                epoch_g_component_sums["tv"] += last_g_items["tv"]
                epoch_g_component_sums["grad"] += last_g_items["grad"]
                epoch_g_component_sums["ms"] += last_g_items["ms"]
                epoch_g_component_sums["color"] += last_g_items["color"]
                epoch_g_raw_component_sums["l1_valid"] += raw_l1_valid
                epoch_g_raw_component_sums["l1_hole"] += raw_l1_hole
                epoch_g_raw_component_sums["perc"] += raw_perc
                epoch_g_raw_component_sums["style"] += raw_style
                epoch_g_raw_component_sums["tv"] += raw_tv
                epoch_g_raw_component_sums["grad"] += raw_grad
                epoch_g_raw_component_sums["ms"] += raw_ms
                epoch_g_raw_component_sums["color"] += raw_color
                epoch_g_component_count += 1

            avg_g = sum(epoch_g_losses[-10:]) / len(epoch_g_losses[-10:]) if epoch_g_losses else 0.0
            avg_d = sum(epoch_d_losses[-10:]) / len(epoch_d_losses[-10:]) if epoch_d_losses else 0.0
            train_pbar.set_description(f"Epoch {epoch + 1}/{args.num_epochs}")
            if use_gan:
                train_pbar.set_postfix({"G": f"{avg_g:.4f}", "D": f"{avg_d:.4f}", "GAN": f"{last_g_items['GAN']:.3f}", "l1v": f"{last_g_items['l1_valid']:.3f}", "l1h": f"{last_g_items['l1_hole']:.3f}", "perc": f"{last_g_items['perc']:.3f}", "style": f"{last_g_items['style']:.3f}", "tv": f"{last_g_items['tv']:.3f}", "grad": f"{last_g_items['grad']:.3f}", "ms": f"{last_g_items['ms']:.3f}", "color": f"{last_g_items['color']:.3f}"})
            else:
                train_pbar.set_postfix({"loss": f"{avg_g:.4f}", "l1v": f"{last_g_items['l1_valid']:.3f}", "l1h": f"{last_g_items['l1_hole']:.3f}", "perc": f"{last_g_items['perc']:.3f}", "style": f"{last_g_items['style']:.3f}", "tv": f"{last_g_items['tv']:.3f}", "grad": f"{last_g_items['grad']:.3f}", "ms": f"{last_g_items['ms']:.3f}", "color": f"{last_g_items['color']:.3f}"})
            train_pbar.update(1)

        scheduler_g.step()
        if use_gan and scheduler_d is not None:
            scheduler_d.step()

        avg_g_loss = sum(epoch_g_losses) / len(epoch_g_losses) if epoch_g_losses else 0.0
        avg_d_loss = sum(epoch_d_losses) / len(epoch_d_losses) if epoch_d_losses else 0.0
        avg_recon = sum(epoch_recon_losses) / len(epoch_recon_losses) if epoch_recon_losses else float("inf")
        avg_g_components = {
            "gan": epoch_g_component_sums["GAN"] / max(epoch_g_component_count, 1),
            "l1_valid": epoch_g_component_sums["l1_valid"] / max(epoch_g_component_count, 1),
            "l1_hole": epoch_g_component_sums["l1_hole"] / max(epoch_g_component_count, 1),
            "perceptual": epoch_g_component_sums["perc"] / max(epoch_g_component_count, 1),
            "style": epoch_g_component_sums["style"] / max(epoch_g_component_count, 1),
            "tv": epoch_g_component_sums["tv"] / max(epoch_g_component_count, 1),
            "grad": epoch_g_component_sums["grad"] / max(epoch_g_component_count, 1),
            "ms_ssim": epoch_g_component_sums["ms"] / max(epoch_g_component_count, 1),
            "color": epoch_g_component_sums["color"] / max(epoch_g_component_count, 1),
        }
        avg_g_raw_components = {
            "raw_l1_valid": epoch_g_raw_component_sums["l1_valid"] / max(epoch_g_component_count, 1),
            "raw_l1_hole": epoch_g_raw_component_sums["l1_hole"] / max(epoch_g_component_count, 1),
            "raw_perceptual": epoch_g_raw_component_sums["perc"] / max(epoch_g_component_count, 1),
            "raw_style": epoch_g_raw_component_sums["style"] / max(epoch_g_component_count, 1),
            "raw_tv": epoch_g_raw_component_sums["tv"] / max(epoch_g_component_count, 1),
            "raw_grad": epoch_g_raw_component_sums["grad"] / max(epoch_g_component_count, 1),
            "raw_ms_ssim": epoch_g_raw_component_sums["ms"] / max(epoch_g_component_count, 1),
            "raw_color": epoch_g_raw_component_sums["color"] / max(epoch_g_component_count, 1),
        }
        avg_d_components = {
            "d_real": epoch_d_component_sums["D_real"] / max(epoch_d_component_count, 1),
            "d_fake": epoch_d_component_sums["D_fake"] / max(epoch_d_component_count, 1),
        }
        epoch_record = {
            "epoch": epoch + 1,
            "g_loss": avg_g_loss,
            "d_loss": avg_d_loss,
            "recon": avg_recon,
            **avg_g_components,
            **avg_g_raw_components,
            **avg_d_components,
        }
        loss_history.append(epoch_record)
        if live_plotter is not None:
            live_plotter(loss_history)

        if use_gan:
            print(f"[Epoch {epoch + 1}] G_loss: {avg_g_loss:.4f}, D_loss: {avg_d_loss:.4f}, Recon: {avg_recon:.4f}")
            print(
                "  G comps(weighted): "
                f"GAN={avg_g_components['gan']:.4f}, "
                f"l1_valid={avg_g_components['l1_valid']:.4f}, "
                f"l1_hole={avg_g_components['l1_hole']:.4f}, "
                f"perc={avg_g_components['perceptual']:.4f}, "
                f"style={avg_g_components['style']:.4f}, "
                f"tv={avg_g_components['tv']:.4f}, "
                f"grad={avg_g_components['grad']:.4f}, "
                f"ms_ssim={avg_g_components['ms_ssim']:.4f}, "
                f"color={avg_g_components['color']:.4f}"
            )
            print(
                "  G comps(raw): "
                f"l1_valid={avg_g_raw_components['raw_l1_valid']:.4f}, "
                f"l1_hole={avg_g_raw_components['raw_l1_hole']:.4f}, "
                f"perc={avg_g_raw_components['raw_perceptual']:.4f}, "
                f"style={avg_g_raw_components['raw_style']:.4f}, "
                f"tv={avg_g_raw_components['raw_tv']:.4f}, "
                f"grad={avg_g_raw_components['raw_grad']:.4f}, "
                f"ms_ssim={avg_g_raw_components['raw_ms_ssim']:.4f}, "
                f"color={avg_g_raw_components['raw_color']:.4f}"
            )
            print(f"  D comps: D_real={avg_d_components['d_real']:.4f}, D_fake={avg_d_components['d_fake']:.4f}")
        else:
            print(f"[Epoch {epoch + 1}] Loss: {avg_g_loss:.4f}, Recon: {avg_recon:.4f}")
            print(
                "  G comps(weighted): "
                f"l1_valid={avg_g_components['l1_valid']:.4f}, "
                f"l1_hole={avg_g_components['l1_hole']:.4f}, "
                f"perc={avg_g_components['perceptual']:.4f}, "
                f"style={avg_g_components['style']:.4f}, "
                f"tv={avg_g_components['tv']:.4f}, "
                f"grad={avg_g_components['grad']:.4f}, "
                f"ms_ssim={avg_g_components['ms_ssim']:.4f}, "
                f"color={avg_g_components['color']:.4f}"
            )
            print(
                "  G comps(raw): "
                f"l1_valid={avg_g_raw_components['raw_l1_valid']:.4f}, "
                f"l1_hole={avg_g_raw_components['raw_l1_hole']:.4f}, "
                f"perc={avg_g_raw_components['raw_perceptual']:.4f}, "
                f"style={avg_g_raw_components['raw_style']:.4f}, "
                f"tv={avg_g_raw_components['raw_tv']:.4f}, "
                f"grad={avg_g_raw_components['raw_grad']:.4f}, "
                f"ms_ssim={avg_g_raw_components['raw_ms_ssim']:.4f}, "
                f"color={avg_g_raw_components['raw_color']:.4f}"
            )

        recon_improved = avg_recon < best_recon
        if recon_improved:
            best_recon = avg_recon

        payload = {
            "epoch": epoch,
            "generator": generator.state_dict(),
            "opt_g": opt_g.state_dict(),
            "scheduler_g": scheduler_g.state_dict(),
            "loss_history": loss_history,
            "best_recon": best_recon,
            "experiment": MAIN_EXPERIMENT_KEY,
            "config": vars(args),
        }
        if use_gan and discriminator is not None and opt_d is not None and scheduler_d is not None:
            payload["discriminator"] = discriminator.state_dict()
            payload["opt_d"] = opt_d.state_dict()
            payload["scheduler_d"] = scheduler_d.state_dict()

        torch.save(payload, latest_checkpoint)

    save_json(output_root / "train_config.json", vars(args))
    train_pbar.close()
    print(f"Training finished. Latest checkpoint: {latest_checkpoint}")


def load_inference_model(args, device: torch.device):
    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    generator = make_generator(args, pretrained_path=None, device=device)
    load_generator_checkpoint(generator, checkpoint_path, device)
    generator.eval()
    return generator


def notebook_style_inference(
    model: torch.nn.Module,
    test_images_dir: str,
    test_masks_dir: str,
    infer_dir: str,
    target_width: int,
    crop_height: int,
    final_width: int,
    overlap: int,
    device: torch.device,
    output_suffix: str,
) -> None:
    infer_path = Path(infer_dir)
    infer_path.mkdir(parents=True, exist_ok=True)
    image_files = sorted([name for name in os.listdir(test_images_dir) if name.lower().endswith(".png")])
    print(f"Found {len(image_files)} test images")

    for image_name in tqdm(image_files, desc="Notebook-style inference"):
        img_path = os.path.join(test_images_dir, image_name)
        mask_path = os.path.join(test_masks_dir, f"{Path(image_name).stem}_mask.png")
        if not os.path.exists(mask_path):
            continue
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        nat = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None or nat is None:
            continue

        img = scale_width(img, target_width)
        nat = scale_width(nat, target_width)
        content_width = img.shape[1]
        img = pad_to_width(img, final_width, False)
        nat = pad_to_width(nat, final_width, True)
        proc_height = img.shape[0]
        nat = (nat > 127).astype(np.uint8) * 255
        nat = cv2.dilate(nat, np.ones((1, 3), np.uint8), iterations=1)

        img_t = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)).float() / 255.0
        nat_t = torch.from_numpy(nat.astype(np.float32) / 255.0).unsqueeze(0)
        restored_patches = []
        crop_indices = list(vertical_crops(img, crop_height, overlap))
        with torch.no_grad():
            for y0, y1 in crop_indices:
                img_patch = img_t[:, y0:y1, :]
                nat_patch = nat_t[:, y0:y1, :]
                if img_patch.shape[1] != crop_height:
                    pad_bottom = crop_height - img_patch.shape[1]
                    img_patch = F.pad(img_patch.unsqueeze(0), (0, 0, 0, pad_bottom), mode="reflect").squeeze(0)
                    nat_patch = F.pad(nat_patch.unsqueeze(0), (0, 0, 0, pad_bottom), mode="reflect").squeeze(0)

                img_patch = img_patch.unsqueeze(0).to(device)
                nat_patch = nat_patch.unsqueeze(0).to(device)
                pred, _ = model(img_patch, nat_patch, None, None)
                pred = pred.clamp(0.0, 1.0)
                hole = (nat_patch > 0.5).float()
                restored_patches.append((pred * hole + img_patch * (1.0 - hole)).squeeze(0).cpu())

        restored_image = torch.zeros(3, proc_height, final_width)
        weight_map = torch.zeros(1, proc_height, final_width)
        for idx, (y0, y1) in enumerate(crop_indices):
            patch = restored_patches[idx]
            actual_height = min(y1, proc_height) - y0
            weights = torch.ones(1, actual_height, final_width)
            if idx > 0:
                fade_in = min(overlap, actual_height)
                for row in range(fade_in):
                    weights[0, row, :] = row / max(fade_in, 1)
            if idx < len(crop_indices) - 1:
                fade_out = min(overlap, actual_height)
                for row in range(fade_out):
                    weights[0, actual_height - 1 - row, :] *= row / max(fade_out, 1)
            restored_image[:, y0 : y0 + actual_height, :] += patch[:, :actual_height, :] * weights
            weight_map[:, y0 : y0 + actual_height, :] += weights

        restored_image = restored_image / weight_map.clamp(min=1e-8)
        result = (restored_image.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        if final_width != content_width:
            if content_width <= final_width:
                left = max(0, (final_width - content_width) // 2)
                result = result[:, left : left + content_width]
            else:
                result = cv2.resize(result, (content_width, proc_height))
        cv2.imwrite(str(infer_path / f"{Path(image_name).stem}_{output_suffix}.png"), result)

    print(f"Inference finished. Results saved to: {infer_path}")


def infer_main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(build_infer_parser(), argv)
    set_seed(args.seed)
    device = resolve_device(args.device_index)
    model = load_inference_model(args, device)
    notebook_style_inference(
        model=model,
        test_images_dir=args.test_images_dir,
        test_masks_dir=args.test_masks_dir,
        infer_dir=args.infer_dir,
        target_width=args.target_width,
        crop_height=args.crop_height,
        final_width=args.final_width,
        overlap=args.overlap,
        device=device,
        output_suffix=args.output_suffix,
    )
