"""Central config for RETFound LoRA age experiments."""

from pathlib import Path

# Base project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"

# Dataset paths
CSV_PATH = PROJECT_ROOT / "metadata/image_age_mapping.csv"
BACKBONE_CKPT = PROJECT_ROOT / "RETFound_MAE_Model/RETFound_mae_natureOCT.pth"

# Optional: explicit group folders if you use them elsewhere
GROUND_FOLDER = DATA_ROOT / "Controls"
HLS_FOLDER = DATA_ROOT / "HLS"
RECOVERY_FOLDER = DATA_ROOT / "Recovery"

# Image settings
IMG_SIZE = 224  # image resolution
IMAGE_TYPES = ["BScanThumb", "REGAVG"]
DAY_WHITELIST = [0, 90]
# Simple experiment: limit to cohorts 1 & 2
COHORTS_TO_KEEP = ["1", "2" , "3"]
COHORT4_GROUP_AS = "High_CO2"  # non-training group label for cohort 4 (to exclude from Controls)

# LoRA hyperparameters
LORA_RANK = 16
LORA_BLOCKS =8
LORA_ALPHA = 16.0
LORA_DROPOUT = 0.20
UPSAMPLE_FACTOR = 4

# Training hyperparameters
BATCH_SIZE = 16  # adjust at runtime if VRAM is tight
NUM_WORKERS = 4
EPOCHS = 40
LR = 3e-4
VAL_SPLIT = 0.2
TEST_SPLIT = 0.0

# Group selection
TRAIN_GROUPS = ["Controls"]
TEST_GROUPS = ["HLS (U)"]

# Optional leave-one-day-out evaluation
HOLDOUT_DAY = None  # e.g., 97 to remove from train/val; None disables holdout
HOLDOUT_TEST_ONLY = False  # if True, test loader will be limited to the holdout day

# Optional data-efficiency subsetting
SUBSET_SIZE = None
SUBSET_FRACTION = None

# Augmentation defaults
AUG_LEVEL = "medium"  # low | medium | high
MIXUP_ALPHA = 0.2   # uniform mix ratios to bridge sparse days
MIXUP_PROB = 0.2
LABEL_NOISE_STD = 2.0  # small biological jitter around nominal day labels
CUTMIX_ALPHA = 0.0
CUTMIX_PROB = 0.0
