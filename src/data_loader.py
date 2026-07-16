import csv
import json
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image

from config import (
    TRAIN_LABELS_CSV, CLASS_MAPPING_JSON, BATCH_SIZE, NUM_WORKERS,
)
from augmentations import get_train_transforms, get_eval_transforms


class PlantVillageDataset(Dataset):
    def __init__(self, split: str, transform=None, labels_csv=TRAIN_LABELS_CSV):
        assert split in {"train", "val", "test"}, f"Invalid split: {split}"

        with open(labels_csv) as f:
            all_rows = list(csv.DictReader(f))

        self.rows = [r for r in all_rows if r["split"] == split]
        if len(self.rows) == 0:
            raise ValueError(f"No rows found for split='{split}' in {labels_csv}")

        self.transform = transform
        self.split = split

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img_path = row["filepath"]
        label = int(row["class_idx"])

        image = Image.open(img_path).convert("RGB")  # guards against grayscale/CMYK stragglers

        if self.transform:
            image = self.transform(image)

        return image, label

    def class_counts(self) -> Counter:
        return Counter(int(r["class_idx"]) for r in self.rows)


def build_weighted_sampler(dataset: PlantVillageDataset) -> WeightedRandomSampler:
    counts = dataset.class_counts()
    class_weights = {cls_idx: 1.0 / count for cls_idx, count in counts.items()}

    sample_weights = [
        class_weights[int(row["class_idx"])] for row in dataset.rows
    ]
    sample_weights = torch.DoubleTensor(sample_weights)

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def get_dataloaders(batch_size: int = BATCH_SIZE, num_workers: int = NUM_WORKERS):
    with open(CLASS_MAPPING_JSON) as f:
        class_mapping = json.load(f)

    train_ds = PlantVillageDataset("train", transform=get_train_transforms())
    val_ds = PlantVillageDataset("val", transform=get_eval_transforms())
    test_ds = PlantVillageDataset("test", transform=get_eval_transforms())

    train_sampler = build_weighted_sampler(train_ds)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader, class_mapping


if __name__ == "__main__":
    # Quick sanity check: confirm batches are shaped correctly and that the
    # weighted sampler is actually flattening class frequency in practice.
    train_loader, val_loader, test_loader, class_mapping = get_dataloaders()

    print(f"Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}  |  Test batches: {len(test_loader)}")

    images, labels = next(iter(train_loader))
    print(f"Batch image tensor shape: {images.shape}")  # [B, 3, IMG_SIZE, IMG_SIZE]
    print(f"Batch label tensor shape: {labels.shape}")

    # Sample a few batches and check class spread — with weighting, no single
    # class should dominate the way it would in the raw ~36:1 distribution.
    seen = Counter()
    for i, (_, labels) in enumerate(train_loader):
        seen.update(labels.tolist())
        if i >= 20:
            break
    print(f"\nClass spread across 20 sampled batches (should look roughly even):")
    print(seen.most_common(5), "...", seen.most_common()[-5:])