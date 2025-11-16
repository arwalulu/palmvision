# src/evaluation/preview_grid.py
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf

from src.configs.loader import load_config
from src.data.tfdata import build_datasets

def save_grid(images, labels, idx_to_class, out_path: Path, cols=8):
    n = min(images.shape[0], 32)
    rows = (n + cols - 1) // cols
    plt.figure(figsize=(cols*2, rows*2))
    for i in range(n):
        ax = plt.subplot(rows, cols, i+1)
        ax.imshow(images[i].numpy())
        ax.set_title(idx_to_class[int(labels[i].numpy())], fontsize=8)
        ax.axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def main():
    cfg = load_config()
    train_ds, _, _, idx_to_class = build_datasets(cfg)
    # take one batch
    for imgs, labs in train_ds.take(1):
        out = Path("experiments/reports/train_batch_grid.jpg")
        save_grid(imgs, labs, idx_to_class, out)
        print(f"[INFO] Saved grid → {out.resolve()}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
