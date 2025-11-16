# src/models/summary.py
import sys, tensorflow as tf
from src.configs.loader import load_config
from src.models.effb0_cbam import build_effb0_cbam
from src.data.tfdata import build_datasets

def main():
    cfg = load_config()
    model = build_effb0_cbam(cfg)
    model.summary(line_length=120)

    # fetch one batch to confirm forward pass
    train_ds, _, _, _ = build_datasets(cfg)
    imgs, labs = next(iter(train_ds))
    out = model(imgs[:4])  # small batch
    print("Forward OK. out shape:", out.shape)
    return 0

if __name__ == "__main__":
    sys.exit(main())
