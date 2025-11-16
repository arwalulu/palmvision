# src/data/tfdata.py
import tensorflow as tf
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple

from src.configs.loader import load_config
from src.stages.resize_label.transforms_tf import decode_and_resize, augment

AUTOTUNE = tf.data.AUTOTUNE

def _read_split_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Split CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if "path" not in df.columns or "label" not in df.columns:
        raise ValueError(f"CSV must have columns [path,label]: {csv_path}")
    return df

def _build_label_maps(classes):
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    return class_to_idx, idx_to_class

def _make_dataset(df: pd.DataFrame, class_to_idx: Dict[str,int], img_size: int,
                  batch_size: int, shuffle: bool, augment_cfg: dict) -> tf.data.Dataset:
    paths = df["path"].tolist()
    labels = [class_to_idx[l] for l in df["label"].tolist()]

    ds_x = tf.data.Dataset.from_tensor_slices(paths)
    ds_y = tf.data.Dataset.from_tensor_slices(labels)

    ds = tf.data.Dataset.zip((ds_x, ds_y))

    def _load_map(p, y):
        img = decode_and_resize(p, img_size)
        return img, tf.cast(y, tf.int32)

    ds = ds.map(_load_map, num_parallel_calls=AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(paths), 4096), reshuffle_each_iteration=True)

    # augmentation only for train
    if augment_cfg and shuffle:
        def _aug_map(img, y):
            img = augment(img, augment_cfg)
            return img, y
        ds = ds.map(_aug_map, num_parallel_calls=AUTOTUNE)

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds

def build_datasets(cfg=None) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, Dict[int,str]]:
    """
    Returns: train_ds, val_ds, test_ds, idx_to_class
    """
    if cfg is None:
        cfg = load_config()

    splits_dir = Path("manifests") / "splits"
    train_csv = splits_dir / "train.csv"
    val_csv   = splits_dir / "val.csv"
    test_csv  = splits_dir / "test.csv"

    df_train = _read_split_csv(train_csv)
    df_val   = _read_split_csv(val_csv)
    df_test  = _read_split_csv(test_csv)

    class_to_idx, idx_to_class = _build_label_maps(cfg.classes)

    img_size   = int(cfg.training["img_size"])
    batch_size = int(cfg.training["batch_size"])
    aug_cfg    = cfg.training.get("augment", {"flip": True, "rotate_deg": 0, "color_jitter": False})

    train_ds = _make_dataset(df_train, class_to_idx, img_size, batch_size, shuffle=True,  augment_cfg=aug_cfg)
    val_ds   = _make_dataset(df_val,   class_to_idx, img_size, batch_size, shuffle=False, augment_cfg=None)
    test_ds  = _make_dataset(df_test,  class_to_idx, img_size, batch_size, shuffle=False, augment_cfg=None)

    return train_ds, val_ds, test_ds, idx_to_class
