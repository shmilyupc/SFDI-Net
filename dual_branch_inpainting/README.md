# Dual-Branch Inpainting

This package is a cleaned and reorganized copy of the dual-branch inpainting
workflow. The code is grouped by responsibility and prepared as a minimal
`train + infer` project.

## Project Layout

```text
dual_branch_inpainting/
├── data/
│   ├── train_images_with_gaps/
│   ├── train_masks/
│   ├── test_images/
│   └── test_masks/
├── dual_branch_inpainting/
│   ├── cli/
│   ├── data/
│   ├── losses/
│   ├── models/
│   ├── experiments.py
│   ├── factory.py
│   └── workflow.py
├── notebooks/
│   └── dual_branch_workflow.ipynb
├── outputs/
├── requirements.txt
└── README.md
```

## Package Structure

- `dual_branch_inpainting/models/`: generator, discriminator, and PConv building blocks
- `dual_branch_inpainting/losses/`: adversarial and inpainting losses
- `dual_branch_inpainting/data/`: datasets and synthetic mask generation
- `dual_branch_inpainting/workflow.py`: train and infer entry functions
- `dual_branch_inpainting/cli/`: command-line wrappers
- `notebooks/`: interactive workflow notebook

## Setup

```bash
cd dual_branch_inpainting
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If your `torch` build is managed separately, install `torch` and `torchvision`
first, then install the remaining packages from `requirements.txt`.

## Data Preparation

Put your data into these folders:

- `data/train_images_with_gaps/`: paired training images with missing regions
- `data/train_masks/`: masks for `train_images_with_gaps`, named `xxx_mask.png`
- `data/test_images/`: inference input images
- `data/test_masks/`: inference masks, named `xxx_mask.png`

Training uses paired image-mask data only. The package no longer includes the
previous complete-image training path or the old full-image evaluation path.

The packaged defaults point to these relative paths, so there are no machine-
specific `/home/...` dependencies left in the code.

## Usage

Notebook:

```bash
jupyter notebook notebooks/dual_branch_workflow.ipynb
```

CLI training:

```bash
python -m dual_branch_inpainting.cli.train --device-index 0
```

CLI inference:

```bash
python -m dual_branch_inpainting.cli.infer --device-index 0 --checkpoint-path outputs/full_model/checkpoints/latest.pt
```

Useful training override example:

```bash
python -m dual_branch_inpainting.cli.train \
  --images-dir data/train_images_with_gaps \
  --masks-dir data/train_masks \
  --output-root outputs/full_model
```

## Notes

- `outputs/` stores checkpoints, logs, and inference images.
- Training starts from scratch by default because no pretrained checkpoint is
  bundled with this package.
- `dual_branch_inpainting/losses/inpainting.py` uses VGG16 perceptual features
  via `torchvision`; the first run may require cached pretrained weights.
