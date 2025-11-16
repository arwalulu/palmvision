# src/data/pipeline.py
import csv, random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict

from sklearn.model_selection import train_test_split

@dataclass
class Sample:
    path: str
    label: str

class DatasetPipeline:
    """
    Reads normalized set (or its manifest) and prepares stratified train/val/test splits.
    Saves split manifests to manifests/splits/.
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.root = Path(".").resolve()
        self.norm_dir = self.root / cfg.dirs.normalized
        self.clean_dir = self.root / cfg.dirs.clean
        self.manifest_norm = self.root / cfg.manifests.normalized
        self.splits_dir = self.root / "manifests" / "splits"
        self.splits_dir.mkdir(parents=True, exist_ok=True)
        self.classes = list(cfg.classes)

    # ---------- load ----------
    def _load_from_manifest(self) -> List[Sample]:
        if not self.manifest_norm.exists():
            raise FileNotFoundError(f"Normalized manifest not found: {self.manifest_norm}")
        rows = []
        with self.manifest_norm.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r["status"] != "normalized":     # skip failed rows, if any
                    continue
                rows.append(Sample(path=r["dst_path"], label=r["class"]))
        return rows

    def _scan_fs(self) -> List[Sample]:
        # fallback if manifest missing (not expected for you)
        rows = []
        for c in self.classes:
            cdir = self.norm_dir / c
            if not cdir.exists(): 
                continue
            for p in cdir.iterdir():
                if p.is_file():
                    rows.append(Sample(path=str(p), label=c))
        return rows

    def load_samples(self) -> List[Sample]:
        try:
            samples = self._load_from_manifest()
        except FileNotFoundError:
            samples = self._scan_fs()
        # sanity: keep only classes we care about
        samples = [s for s in samples if s.label in self.classes]
        if not samples:
            raise RuntimeError("No samples found in normalized set.")
        return samples

    # ---------- split ----------
    def split(self, samples: List[Sample]) -> Tuple[List[Sample], List[Sample], List[Sample]]:
        val_ratio = float(self.cfg.splits["val"])
        test_ratio = float(self.cfg.splits["test"])
        seed = int(self.cfg.seed)

        X = [s.path for s in samples]
        y = [s.label for s in samples]

        # first split off val+test together
        X_train, X_tmp, y_train, y_tmp = train_test_split(
            X, y, test_size=val_ratio + test_ratio, stratify=y, random_state=seed
        )
        rel_test = test_ratio / (val_ratio + test_ratio) if (val_ratio + test_ratio) > 0 else 0.0
        X_val, X_test, y_val, y_test = train_test_split(
            X_tmp, y_tmp, test_size=rel_test, stratify=y_tmp, random_state=seed
        )

        to_samples = lambda Xs, ys: [Sample(path=p, label=l) for p, l in zip(Xs, ys)]
        return to_samples(X_train, y_train), to_samples(X_val, y_val), to_samples(X_test, y_test)

    # ---------- save ----------
    def _write_manifest(self, name: str, rows: List[Sample]) -> Path:
        out = self.splits_dir / name
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["path","label"])
            writer.writeheader()
            for s in rows:
                writer.writerow({"path": s.path, "label": s.label})
        return out

    def save_splits(self, train: List[Sample], val: List[Sample], test: List[Sample]) -> Dict[str, Path]:
        out = {
            "train": self._write_manifest("train.csv", train),
            "val":   self._write_manifest("val.csv",   val),
            "test":  self._write_manifest("test.csv",  test),
        }
        return out

    # ---------- quick counts ----------
    @staticmethod
    def class_counts(rows: List[Sample]) -> Dict[str, int]:
        c = {}
        for r in rows: c[r.label] = c.get(r.label, 0) + 1
        return c
