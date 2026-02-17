# Vision Transformer from Scratch

A clean, modular implementation of the **Vision Transformer (ViT)** in TensorFlow/Keras, following the architecture in [*An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*](https://arxiv.org/abs/2010.11929) (Dosovitskiy et al., ICLR 2021).

Suitable for learning, experimentation, and as a base for custom variants. Includes training on **CIFAR-10** with optional **TPU/GPU** support.

---

## Features

- **Patch embedding** with learnable [CLS] token and positional embeddings  
- **Transformer encoder** with pre-norm multi-head self-attention and MLP blocks  
- **ViT classifier** with configurable depth, width, and patch size  
- **Data pipeline**: normalization, resizing, augmentation (flip, rotation, zoom)  
- **Training script** with YAML config, checkpointing, and TPU/GPU strategy selection  
- **Notebook** for step-by-step exploration (patch extraction, layers, training)

---

## Project Structure

```
Vision-Transformers-from-scratch/
├── README.md
├── LICENSE
├── requirements.txt
├── pyproject.toml
├── configs/
│   └── default.yaml          # Training and model config
├── src/
│   └── vit/
│       ├── __init__.py
│       ├── layers.py         # PatchEmbedding, TransformerLayer
│       ├── models.py         # TransformerEncoder, ViT
│       ├── data.py           # Preprocessing, augmentation, datasets
│       └── train.py          # Strategy + model build/compile helpers
├── scripts/
│   └── train.py              # CLI training on CIFAR-10
├── notebooks/
│   └── vision_transformers.ipynb
└── tests/
    └── test_models.py
```

---

## Setup

**Requirements:** Python 3.9+, TensorFlow 2.12+

```bash
cd Vision-Transformers-from-scratch
python -m venv .venv
source .venv/bin/activate   # or: .venv\Scripts\activate on Windows
pip install -r requirements.txt
pip install -e .            # install vit package in editable mode
```

---

## Quick Start

**Train on CPU/GPU (CIFAR-10, config from `configs/default.yaml`):**

```bash
python scripts/train.py
```

**Override epochs and batch size:**

```bash
python scripts/train.py --epochs 10 --batch-size 512
```

**Use TPU (e.g. on Colab):**

```bash
python scripts/train.py --use-tpu --tpu-address ""
```

**Custom config file:**

```bash
python scripts/train.py --config configs/default.yaml --save-dir my_checkpoints
```

Checkpoints are saved under `checkpoints/` (or `--save-dir`): `vit_best.keras` (best val accuracy) and `vit_final.keras`.

---

## Configuration

Edit `configs/default.yaml` to change:

- **dataset:** `image_size`, `num_classes`
- **model:** `patch_size`, `d_model`, `num_heads`, `num_layers`, `mlp_ratio`, `dropout_rate`
- **training:** `batch_size`, `epochs`, `use_tpu`, `tpu_address`

`num_patches` is derived as `(image_size // patch_size) ** 2`; for 72×72 and `patch_size: 6` you get 144 patches.

---

## Using the Package

```python
from vit import ViT, PatchEmbedding, TransformerLayer
from vit.data import load_cifar10, build_preprocessing, create_dataset, build_augmentation
from vit.train import get_strategy, build_and_compile_vit

# Build model
model = ViT(
    num_classes=10,
    patch_size=6,
    num_patches=(72 // 6) ** 2,
    d_model=128,
    num_heads=2,
    num_layers=4,
    mlp_ratio=2.0,
    dropout_rate=0.1,
)
model.build((None, 72, 72, 3))
model.summary()
```

---

## Notebook

Open `notebooks/vision_transformers.ipynb` for:

1. Loading an image and extracting patches  
2. Visualizing patches  
3. Defining `PatchEmbedding`, `TransformerLayer`, `TransformerEncoder`, and `ViT`  
4. Training on CIFAR-10 with preprocessing and optional TPU  

Run from repo root so that the `vit` package (and `configs/`) are on the path, or set `PYTHONPATH=src` and use the notebook from the repo root.

---

## Tests

From repo root, either install the package and run pytest, or run with `PYTHONPATH=src`:

```bash
pip install -e ".[dev]"   # optional: installs pytest
pytest tests/ -v

# Or without installing:
PYTHONPATH=src python -m pytest tests/ -v
```

---

## Reference

- **Paper:** [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)  
- **Authors:** Alexey Dosovitskiy, Lucas Beyer, et al. (Google Research)  
- **Code:** This repo is an independent implementation for educational use.

---

## License

MIT. See [LICENSE](LICENSE).
