import json
import csv
import random
from pathlib import Path
from collections import defaultdict

from config import (
    RAW_DATA_DIR, TRAIN_LABELS_CSV, CLASS_MAPPING_JSON,
    CLASS_DISTRIBUTION_CSV, TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    SEED, VALID_EXTENSIONS, ensure_dirs,
)


def list_class_folders(raw_dir: Path) -> list[str]:
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Raw data directory not found: {raw_dir}\n"
            f"Check that your dataset sits at this exact path."
        )
    return sorted([p.name for p in raw_dir.iterdir() if p.is_dir()])


def list_images(class_dir: Path) -> list[Path]:
    return sorted([
        p for p in class_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    ])


def stratified_split_per_class(files: list[Path], seed: int):
    rng = random.Random(seed)
    shuffled = files[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    # remainder goes to test, avoids rounding loss

    train_files = shuffled[:n_train]
    val_files = shuffled[n_train:n_train + n_val]
    test_files = shuffled[n_train + n_val:]

    return train_files, val_files, test_files


def build_split(raw_dir: Path, seed: int):
    class_names = list_class_folders(raw_dir)
    class_mapping = {name: idx for idx, name in enumerate(class_names)}

    rows = []
    class_counts = defaultdict(lambda: {"train": 0, "val": 0, "test": 0, "total": 0})

    for class_name in class_names:
        class_idx = class_mapping[class_name]
        class_dir = raw_dir / class_name
        files = list_images(class_dir)

        if len(files) == 0:
            print(f"  WARNING: no images found in '{class_name}', skipping.")
            continue

        train_files, val_files, test_files = stratified_split_per_class(files, seed)

        for split_name, split_files in [
            ("train", train_files), ("val", val_files), ("test", test_files)
        ]:
            for fpath in split_files:
                rows.append({
                    "filepath": str(fpath),
                    "class_name": class_name,
                    "class_idx": class_idx,
                    "split": split_name,
                })
            class_counts[class_name][split_name] = len(split_files)
            class_counts[class_name]["total"] += len(split_files)

    return rows, class_mapping, class_counts


def write_outputs(rows, class_mapping, class_counts):
    # train_labels.csv — the single source of truth for all splits
    with open(TRAIN_LABELS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filepath", "class_name", "class_idx", "split"])
        writer.writeheader()
        writer.writerows(rows)

    # class_mapping.json — class_name -> class_idx, frozen and reusable
    with open(CLASS_MAPPING_JSON, "w") as f:
        json.dump(class_mapping, f, indent=2)

    # class_distribution.csv — per-class counts per split, feeds EDA + weighted loss
    with open(CLASS_DISTRIBUTION_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["class_name", "class_idx", "train", "val", "test", "total"])
        writer.writeheader()
        for class_name, counts in sorted(class_counts.items()):
            writer.writerow({
                "class_name": class_name,
                "class_idx": class_mapping[class_name],
                **counts,
            })


def main():
    ensure_dirs()
    print(f"Scanning: {RAW_DATA_DIR}")

    rows, class_mapping, class_counts = build_split(RAW_DATA_DIR, SEED)

    write_outputs(rows, class_mapping, class_counts)

    total_images = len(rows)
    n_train = sum(1 for r in rows if r["split"] == "train")
    n_val = sum(1 for r in rows if r["split"] == "val")
    n_test = sum(1 for r in rows if r["split"] == "test")

    print(f"\nClasses found       : {len(class_mapping)}")
    print(f"Total images indexed: {total_images}")
    print(f"  train: {n_train}  ({n_train/total_images:.1%})")
    print(f"  val  : {n_val}  ({n_val/total_images:.1%})")
    print(f"  test : {n_test}  ({n_test/total_images:.1%})")
    print(f"\nWritten:")
    print(f"  {TRAIN_LABELS_CSV}")
    print(f"  {CLASS_MAPPING_JSON}")
    print(f"  {CLASS_DISTRIBUTION_CSV}")


if __name__ == "__main__":
    main()