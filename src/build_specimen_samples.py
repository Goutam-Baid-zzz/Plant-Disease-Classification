import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent / "src"))

from config import RAW_DATA_DIR, SAMPLES_DIR
import random
from PIL import Image

SAMPLES_PER_CLASS = 6
THUMB_SIZE = (300, 300)
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

for class_dir in sorted(RAW_DATA_DIR.iterdir()):
    if not class_dir.is_dir():
        continue
    images = list(class_dir.glob("*.[jJ][pP][gG]")) + list(class_dir.glob("*.[pP][nN][gG]"))
    if not images:
        continue
    chosen = random.sample(images, min(SAMPLES_PER_CLASS, len(images)))

    out_dir = SAMPLES_DIR / class_dir.name
    out_dir.mkdir(exist_ok=True)
    for img_path in chosen:
        img = Image.open(img_path).convert("RGB")
        img.thumbnail(THUMB_SIZE)
        img.save(out_dir / img_path.name, quality=75, optimize=True)

    print(f"{class_dir.name}: saved {len(chosen)} samples")