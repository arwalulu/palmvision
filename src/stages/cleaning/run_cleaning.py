# src/stages/cleaning/run_cleaning.py
import os, sys, csv, shutil, hashlib
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]   # .../palmvision
RAW_DIR      = PROJECT_ROOT / "data" / "raw"
CLEAN_DIR    = PROJECT_ROOT / "data" / "clean"
MANIFESTS    = PROJECT_ROOT / "manifests"
MANIFESTS.mkdir(parents=True, exist_ok=True)
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

# Any weird folder names will be mapped to these canonical class names
CANONICAL_CLASSES = ["Bug", "Dubas", "Healthy", "Honey"]
ALIASES = {
    "bug": "Bug",
    "bugs": "Bug",
    "dubas": "Dubas",
    "dubaas": "Dubas",
    "healthy": "Healthy",
    "honey": "Honey",
    # add more aliases if you find variants
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def canonicalize(name: str) -> str:
    n = name.strip().replace("_", " ").replace("-", " ").lower()
    return ALIASES.get(n, name)  # if exact alias found, map; else keep original as-is

def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTS

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def validate_image(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()  # quick header check
        # re-open to ensure load works (verify() leaves file in indeterminate state)
        with Image.open(path) as im:
            im.load()
        return True
    except (UnidentifiedImageError, OSError):
        return False

def scan_raw_images():
    """Yield tuples of (src_path, raw_class_name) for all image files under RAW_DIR."""
    for root, _, files in os.walk(RAW_DIR):
        root_p = Path(root)
        # class name is assumed to be the last directory in the path under RAW_DIR
        # i.e., .../raw/<class>/file.jpg
        try:
            rel = root_p.relative_to(RAW_DIR)
        except ValueError:
            continue
        parts = rel.parts
        if not parts:
            continue
        raw_class = parts[0]
        for fn in files:
            p = root_p / fn
            if is_image(p):
                yield p, raw_class

def main():
    print(f"[INFO] RAW_DIR   = {RAW_DIR}")
    print(f"[INFO] CLEAN_DIR = {CLEAN_DIR}")
    print(f"[INFO] Writing manifests to: {MANIFESTS}")

    # Prepare output class folders
    for c in CANONICAL_CLASSES:
        (CLEAN_DIR / c).mkdir(parents=True, exist_ok=True)

    # Track duplicates
    seen_hash_global = {}  # hash -> (class, dst_path) first keeper
    seen_hash_per_class = {c: set() for c in CANONICAL_CLASSES}

    rows = []
    kept_counts = {c: 0 for c in CANONICAL_CLASSES}
    corrupt_count = 0
    dup_in_class = 0
    dup_cross = 0

    items = list(scan_raw_images())
    print(f"[INFO] Found {len(items)} candidate image files in raw/")

    for src_path, raw_class in tqdm(items, desc="Cleaning", ncols=100):
        canon = canonicalize(raw_class)
        # If raw class isn’t one of our canonical set, skip but log it
        if canon not in CANONICAL_CLASSES:
            rows.append({
                "status": "skipped_unknown_class",
                "reason": f"raw_class={raw_class}",
                "src_path": str(src_path),
                "dst_path": "",
                "class": raw_class,
                "hash": "",
            })
            continue

        # Validate
        ok = validate_image(src_path)
        if not ok:
            corrupt_count += 1
            rows.append({
                "status": "dropped_corrupt",
                "reason": "PIL verify/load failed",
                "src_path": str(src_path),
                "dst_path": "",
                "class": canon,
                "hash": "",
            })
            continue

        # Hash
        h = sha256_file(src_path)

        # Duplicate within class?
        if h in seen_hash_per_class[canon]:
            dup_in_class += 1
            rows.append({
                "status": "dropped_duplicate_in_class",
                "reason": f"hash={h}",
                "src_path": str(src_path),
                "dst_path": "",
                "class": canon,
                "hash": h,
            })
            continue

        # Duplicate across classes?
        if h in seen_hash_global and seen_hash_global[h][0] != canon:
            dup_cross += 1
            rows.append({
                "status": "dropped_cross_class_duplicate",
                "reason": f"first_seen_in={seen_hash_global[h][0]} hash={h}",
                "src_path": str(src_path),
                "dst_path": "",
                "class": canon,
                "hash": h,
            })
            continue

        # Keep it → copy to clean/<class>/<canonical_name>
        dst_dir = CLEAN_DIR / canon
        dst_dir.mkdir(parents=True, exist_ok=True)
        new_name = f"{canon}_{kept_counts[canon]:06d}{src_path.suffix.lower()}"
        dst_path = dst_dir / new_name
        shutil.copy2(src_path, dst_path)

        kept_counts[canon] += 1
        seen_hash_per_class[canon].add(h)
        seen_hash_global[h] = (canon, dst_path)

        rows.append({
            "status": "kept",
            "reason": "",
            "src_path": str(src_path),
            "dst_path": str(dst_path),
            "class": canon,
            "hash": h,
        })

      # Write manifest (clean)
    clean_manifest = MANIFESTS / "manifest_clean.csv"
    with clean_manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["status","reason","src_path","dst_path","class","hash"])
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    total_kept = sum(kept_counts.values())
    print("\n[SUMMARY]")
    for c, n in kept_counts.items():
        print(f"  Kept {c:7s}: {n}")
    print(f"  Dropped corrupt: {corrupt_count}")
    print(f"  Duplicates in-class: {dup_in_class}")
    print(f"  Duplicates cross-class: {dup_cross}")
    print(f"[INFO] Clean manifest: {clean_manifest}")

if __name__ == "__main__":
    sys.exit(main())


