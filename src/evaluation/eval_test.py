# src/evaluation/eval_test.py
import sys, json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

from src.configs.loader import load_config
from src.data.tfdata import build_datasets
from src.models.cbam import SpatialAttention


def find_latest_run(exp_root: Path) -> Path:
    runs = [d for d in exp_root.glob("run_*") if d.is_dir()]
    if not runs:
        raise FileNotFoundError(f"No run_* directories found in {exp_root}")
    # sort by name (timestamp is in name) or by mtime
    runs = sorted(runs, key=lambda d: d.name)
    return runs[-1]

def load_best_model(run_dir: Path) -> tf.keras.Model:
    ckpt = run_dir / "checkpoints" / "best-val-acc.keras"
    if not ckpt.exists():
        raise FileNotFoundError(f"Best checkpoint not found: {ckpt}")
    print(f"[INFO] Loading best model from: {ckpt}")
    model = tf.keras.models.load_model(
        ckpt,
        compile=False,
        safe_mode=False,
        custom_objects={"SpatialAttention": SpatialAttention},
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )
    return model


def collect_predictions(model: tf.keras.Model, test_ds, idx_to_class: dict[int, str]):
    y_true = []
    y_pred = []

    for imgs, labels in test_ds:
        probs = model.predict(imgs, verbose=0)
        preds = np.argmax(probs, axis=1)
        y_true.append(labels.numpy())
        y_pred.append(preds)

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    # map indices to class names
    class_indices = sorted(idx_to_class.keys())
    class_names = [idx_to_class[i] for i in class_indices]

    return y_true, y_pred, class_names

def plot_confusion_matrix(cm, class_names, out_path: Path):
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix (Test Set)",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # write numbers
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=9,
            )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved confusion matrix → {out_path.resolve()}")

def main():
    cfg = load_config()

    # datasets
    _, _, test_ds, idx_to_class = build_datasets(cfg)

    # pick latest run
    exp_root = Path("experiments")
    run_dir = find_latest_run(exp_root)
    print(f"[INFO] Evaluating run: {run_dir}")

    # load best model
    model = load_best_model(run_dir)

    # evaluate raw metrics
    test_loss, test_acc = model.evaluate(test_ds, verbose=1)
    print(f"\n[TEST] loss={test_loss:.4f}  acc={test_acc:.4f}")

    # predictions & confusion matrix
    y_true, y_pred, class_names = collect_predictions(model, test_ds, idx_to_class)
    cm = confusion_matrix(y_true, y_pred)

    # save confusion matrix plot
    cm_path = run_dir / "test_confusion_matrix.png"
    plot_confusion_matrix(cm, class_names, cm_path)

    # classification report
    report = classification_report(
        y_true, y_pred, target_names=class_names, digits=4
    )
    print("\n[TEST] Classification report:\n")
    print(report)

    # save report
    (run_dir / "test_classification_report.txt").write_text(report, encoding="utf-8")
    print(f"[INFO] Saved classification report → {run_dir/'test_classification_report.txt'}")

    # save simple JSON summary
    summary = {
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "classes": class_names,
    }
    with (run_dir / "test_metrics.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"[INFO] Saved JSON summary → {run_dir/'test_metrics.json'}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
