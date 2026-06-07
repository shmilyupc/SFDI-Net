# SFDI-Net

This repository contains the training and inference code for SFDI-Net.

## Layout

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
│   └── sfdi_net_workflow.ipynb
├── outputs/
├── requirements.txt
└── README.md
```

## Main Files

- `dual_branch_inpainting/models/`: network modules
- `dual_branch_inpainting/losses/`: reconstruction and adversarial losses
- `dual_branch_inpainting/data/`: paired dataset loading and synthetic mask generation
- `dual_branch_inpainting/workflow.py`: training and inference pipeline
- `dual_branch_inpainting/cli/`: command-line entry points
- `notebooks/sfdi_net_workflow.ipynb`: notebook workflow. You can use this Jupyter notebook to quickly get started with training and testing.

## Environment

```bash
cd dual_branch_inpainting
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install the appropriate `torch` and `torchvision` build for your machine before
installing the remaining dependencies if needed.

## Data

Prepare these directories under `data/`:

- `train_images_with_gaps/`
- `train_masks/`
- `test_images/`
- `test_masks/`

Training uses paired image-mask data. Mask filenames should follow the
`xxx_mask.png` convention.

## Usage

Notebook:

```bash
jupyter notebook notebooks/sfdi_net_workflow.ipynb
```

Training:

```bash
python -m dual_branch_inpainting.cli.train --device-index 0
```

Inference:

```bash
python -m dual_branch_inpainting.cli.infer --device-index 0 --checkpoint-path outputs/sfdi_net/checkpoints/latest.pt
```

Example with explicit paths:

```bash
python -m dual_branch_inpainting.cli.train \
  --images-dir data/train_images_with_gaps \
  --masks-dir data/train_masks \
  --output-root outputs/sfdi_net
```

## Notes

- `outputs/` stores checkpoints, logs, and inference results.
- Training starts from scratch unless `--pretrained-unet-path` is provided.
- `dual_branch_inpainting/losses/inpainting.py` uses VGG16 perceptual features
  through `torchvision`.
