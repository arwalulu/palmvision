# src/training/callbacks.py
from pathlib import Path
from tensorflow.keras.callbacks import (ModelCheckpoint, EarlyStopping,
                                        ReduceLROnPlateau, TensorBoard, CSVLogger)

def make_callbacks(run_dir: Path):
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = ckpt_dir / "best-val-acc.keras"  # keeps full model
    logs_dir  = run_dir / "tb"
    logs_dir.mkdir(parents=True, exist_ok=True)

    return [
        ModelCheckpoint(
            filepath=str(ckpt_path),
            monitor="val_acc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_acc",
            mode="max",
            patience=6,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        TensorBoard(
            log_dir=str(logs_dir),
            histogram_freq=0,
            write_graph=False,
            write_images=False,
        ),
        CSVLogger(str(run_dir / "history.csv"))
    ]
