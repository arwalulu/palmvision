# src/data/build_splits.py
import sys
from pathlib import Path
from src.configs.loader import load_config
from src.data.pipeline import DatasetPipeline

def main():
    cfg = load_config()
    dp = DatasetPipeline(cfg)
    samples = dp.load_samples()
    train, val, test = dp.split(samples)
    outs = dp.save_splits(train, val, test)

    # print quick summary
    for split_name, rows in [("train", train), ("val", val), ("test", test)]:
        counts = dp.class_counts(rows)
        total = sum(counts.values())
        pretty = "  ".join([f"{k}:{v}" for k, v in sorted(counts.items())])
        print(f"[{split_name}] total={total}  {pretty}")

    print("[INFO] Split manifests:")
    for k, v in outs.items():
        print(f"  {k} → {v}")

if __name__ == "__main__":
    sys.exit(main())
