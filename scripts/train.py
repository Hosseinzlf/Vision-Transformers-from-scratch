#!/usr/bin/env python3
"""
Train Vision Transformer on CIFAR-10.

Usage (from repo root):
  python scripts/train.py
  python scripts/train.py --config configs/default.yaml --epochs 10
  python scripts/train.py --use-tpu --tpu-address ""
"""

import argparse
import pathlib
import sys
from typing import Optional

# Allow running without installing the package (from repo root)
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import tensorflow as tf
import yaml

from vit.data import (
    build_augmentation,
    build_preprocessing,
    create_dataset,
    load_cifar10,
)
from vit.train import build_and_compile_vit, get_strategy


def load_config(path: pathlib.Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def parse_args():
    p = argparse.ArgumentParser(description="Train ViT on CIFAR-10")
    p.add_argument("--config", type=pathlib.Path, default=pathlib.Path("configs/default.yaml"), help="Config YAML")
    p.add_argument("--epochs", type=int, default=None, help="Override config epochs")
    p.add_argument("--batch-size", type=int, default=None, help="Override config batch size")
    p.add_argument("--use-tpu", action="store_true", help="Use TPU if available")
    p.add_argument("--tpu-address", type=str, default="", help="TPU address (e.g. '' for Colab)")
    p.add_argument("--save-dir", type=pathlib.Path, default=pathlib.Path("checkpoints"), help="Checkpoint directory")
    return p.parse_args()


def main():
    args = parse_args()
    root = _REPO_ROOT
    config_path = root / args.config if not args.config.is_absolute() else args.config
    config = load_config(config_path)

    ds_cfg = config["dataset"]
    model_cfg = config["model"]
    train_cfg = config["training"]

    image_size = tuple(ds_cfg["image_size"])
    num_classes = ds_cfg["num_classes"]
    patch_size = model_cfg["patch_size"]
    batch_size = args.batch_size or train_cfg["batch_size"]
    epochs = args.epochs or train_cfg["epochs"]
    use_tpu = args.use_tpu or train_cfg.get("use_tpu", False)
    tpu_address = args.tpu_address or train_cfg.get("tpu_address", "")

    strategy = get_strategy(use_tpu=use_tpu, tpu_address=tpu_address or None)

    (x_train, y_train), (x_test, y_test) = load_cifar10()
    preprocessing = build_preprocessing(image_size, adapt_data=x_train)
    augmentation = build_augmentation()

    train_ds = create_dataset(
        x_train, y_train,
        batch_size=batch_size,
        preprocessing=preprocessing,
        augmentation=augmentation,
        shuffle=True,
    )
    val_ds = create_dataset(
        x_test, y_test,
        batch_size=batch_size,
        preprocessing=preprocessing,
        shuffle=False,
    )

    with strategy.scope():
        model = build_and_compile_vit(
            num_classes=num_classes,
            image_size=image_size[0],
            patch_size=patch_size,
            d_model=model_cfg["d_model"],
            num_heads=model_cfg["num_heads"],
            num_layers=model_cfg["num_layers"],
            mlp_ratio=model_cfg["mlp_ratio"],
            dropout_rate=model_cfg["dropout_rate"],
        )

    save_dir = root / args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(save_dir / "vit_best.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
        ),
        tf.keras.callbacks.TerminateOnNaN(),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
    )

    model.save(save_dir / "vit_final.keras")
    print(f"Model saved to {save_dir}")


if __name__ == "__main__":
    main()
