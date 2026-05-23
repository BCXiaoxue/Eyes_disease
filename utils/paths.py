from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

TRAIN_IMAGES_DIR = PROJECT_ROOT / "train" / "images"
TEST_IMAGES_DIR = PROJECT_ROOT / "test" / "images"

MODELS_DIR = ARTIFACTS_DIR / "models"
NEW_MODELS_DIR = ARTIFACTS_DIR / "new_models"
FEEDBACK_DIR = ARTIFACTS_DIR / "feedback"
FIGS_DIR = ARTIFACTS_DIR / "figs"
LOGS_DIR = ARTIFACTS_DIR / "logs"

LABEL_CSV = DATA_DIR / "label.csv"
TRAIN_CSV = DATA_DIR / "train.csv"
VAL_CSV = DATA_DIR / "val.csv"
VALIDATION_CSV = DATA_DIR / "validation.csv"
PREDICTED_CSV = DATA_DIR / "predicted.csv"
