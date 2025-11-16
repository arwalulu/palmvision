# src/evaluation/quick_checks.py
import sys, os
from pathlib import Path
import pandas as pd

SPLITS = ["train","val","test"]

def counts_for(csv_path: Path):
    df = pd.read_csv(csv_path)
    return df["label"].value_counts().sort_index()

def main():
    root = Path(".").resolve()
    splits_dir = root / "manifests" / "splits"
    for s in SPLITS:
        csv_path = splits_dir / f"{s}.csv"
        if not csv_path.exists():
            print(f"[WARN] Missing split CSV: {csv_path}")
            continue
        vc = counts_for(csv_path)
        total = vc.sum()
        pretty = "  ".join([f"{idx}:{vc[idx]}" for idx in vc.index])
        print(f"[{s}] total={total}  {pretty}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
