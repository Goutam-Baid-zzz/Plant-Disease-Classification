from pathlib import Path

# ── Project root ─────────────────────────────────────────────
# Assumes this file lives at <PROJECT_ROOT>/src/config.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Data paths ───────────────────────────────────────────────
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "plantvillage dataset"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"
SAMPLES_DIR = PROJECT_ROOT / "assets" / "specimen_samples"

TRAIN_LABELS_CSV = METADATA_DIR / "train_labels.csv"
CLASS_MAPPING_JSON = METADATA_DIR / "class_mapping.json"
CLASS_DISTRIBUTION_CSV = METADATA_DIR / "class_distribution.csv"

# ── Output paths ─────────────────────────────────────────────
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LOGS_DIR = OUTPUTS_DIR / "logs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
GRADCAM_DIR = OUTPUTS_DIR / "gradcam"
MISCLASSIFIED_DIR = OUTPUTS_DIR / "misclassified"

# ── Reproducibility ──────────────────────────────────────────
SEED = 42

# ── Split ratios (must sum to 1.0) ───────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ── Image / training settings ────────────────────────────────
IMG_SIZE = 128          
BATCH_SIZE = 32
NUM_WORKERS = 4
NORMALIZE_MEAN = [0.5, 0.5, 0.5]
NORMALIZE_STD = [0.5, 0.5, 0.5]

# Valid image extensions to scan for in raw data
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}

LEARNING_RATE = 1e-3
BASELINE_EPOCHS = 15
IMPROVED_EPOCHS = 25
EARLY_STOPPING_PATIENCE = 5  


def ensure_dirs():
    for d in [
        PROCESSED_DATA_DIR, METADATA_DIR, MODELS_DIR,
        LOGS_DIR, PLOTS_DIR, GRADCAM_DIR, MISCLASSIFIED_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_dirs()
    print(f"PROJECT_ROOT      : {PROJECT_ROOT}")
    print(f"RAW_DATA_DIR       : {RAW_DATA_DIR}")
    print(f"RAW_DATA_DIR exists: {RAW_DATA_DIR.exists()}")