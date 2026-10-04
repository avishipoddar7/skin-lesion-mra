"""All tunable parameters for the project live here (PRD NFR-05)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PH2_DIR = ROOT / "data" / "PH2"
PH2_IMAGES = PH2_DIR / "PH2 Dataset images"
PH2_LABELS = PH2_DIR / "PH2_dataset.txt"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
SAMPLES_DIR = ROOT / "assets" / "samples"
MODEL_PATH = MODELS_DIR / "model.pkl"

SEED = 42
IMG_SIZE = 256

# Default (final) configuration. Fixed in advance, not chosen from the grid.
WAVELET = "haar"
LEVEL = 2
HAIR_REMOVAL = True

WAVELETS = ("haar", "db4", "sym4")
LEVELS = (1, 2, 3)

# Hair removal
HAIR_KERNEL = 11
HAIR_THRESHOLD = 25  # tuned: the PRD's 10 flagged ~34% of pixels on PH2
HAIR_MIN_AREA = 40   # drop specks smaller than this (px)
HAIR_DILATE = 3
INPAINT_RADIUS = 5

# Segmentation (tuned once on PH2 for the default config, then frozen)
BLUR_KSIZE = 5
OPEN_SIZE = 5
CLOSE_SIZE = 15
BORDER_PENALTY = 0.25
VIGNETTE_RATIO = 0.40  # border-touching pixels darker than this x median are vignette
VIGNETTE_GROW = 7      # dilate the vignette mask by this many px before filling
MIN_LESION_FRAC = 0.01
MAX_LESION_FRAC = 0.90

# Features
CROP_SIZE = 128

# Classifier
SVM_C = 1.0
CV_FOLDS = 5
# Flag melanoma when its estimated probability reaches the class prevalence (40/200).
# Chosen on principle for a screening setting, not tuned on the results.
DECISION_THRESHOLD = 0.20

# App validation
ALLOWED_EXT = (".jpg", ".jpeg", ".png", ".bmp")
MIN_SIDE = 128
MAX_BYTES = 10 * 1024 * 1024
