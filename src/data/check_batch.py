# src/data/check_batch.py
import sys, tensorflow as tf
from src.configs.loader import load_config
from src.data.tfdata import build_datasets

def main():
    cfg = load_config()
    train_ds, val_ds, test_ds, idx_to_class = build_datasets(cfg)

    # Fetch one batch
    for imgs, labels in train_ds.take(1):
        print("Batch imgs:", imgs.shape, imgs.dtype)
        print("Batch labels:", labels.shape, labels.dtype)
        print("Classes:", idx_to_class)
    return 0

if __name__ == "__main__":
    sys.exit(main())
