"""
Configuration file for brain MRI classification training

This module handles all configuration settings including:
- Command-line argument parsing
- Data paths and cohort settings
- Model architecture parameters
- Training hyperparameters
- Explainability and visualization settings

Usage:
    python train.py -c label -m lora -g cuda:0
"""

import argparse
import os
from pathlib import Path
from typing import Optional, List, Dict


# ============================================================================
# COMMAND-LINE ARGUMENTS
# ============================================================================

parser = argparse.ArgumentParser(description="Brain MRI Classification Training")
parser.add_argument(
    "-c",
    "--column",
    type=str,
    default=None,
    help="Target column for training (default: label)",
)
parser.add_argument(
    "-m",
    "--mode",
    type=str,
    default="sfcn",
    choices=["sfcn", "dense", "linear", "ssl-finetuned", "lora"],
    help="Training mode (default: lora)",
)
parser.add_argument(
    "-g",
    "--gpu",
    type=str,
    default=None,
    help='GPU device (e.g., "cuda:0", default: cuda:0)',
)
parser.add_argument(
    "-i",
    "--eid",
    type=str,
    nargs="*",
    default=None,
    help="Create heatmaps for selected eid",
)
parser.add_argument("--testcohort", type=str, default=None, help="select  cohort")
parser.add_argument("--traincohort", type=str, default=None, help="select cohort")


args = parser.parse_args()

EIDS = args.eid
# ============================================================================
# BASIC SETTINGS
# ============================================================================

IMG_SIZE = 96

# Task Configuration
COLUMN_NAME = args.column if args.column else "label"
TRAINING_MODE = args.mode if args.mode else "sfcn"
TASK = "classification"

# Cohort Configuration
TRAIN_COHORT = "mspaths/t1w"
TEST_COHORT = "mspaths2/t1w"

CSV_NAME_TRAIN = f"{COLUMN_NAME}"
CSV_NAME_TEST = f"{COLUMN_NAME}"

# ============================================================================
# DATA PATHS
# ============================================================================

# CSV Files
CSV_FULL = f"../../data/{TRAIN_COHORT}/{CSV_NAME_TRAIN}.csv"
CSV_TRAIN = f"../../data/{TRAIN_COHORT}/train/{CSV_NAME_TRAIN}.csv"
CSV_VAL = f"../../data/{TRAIN_COHORT}/val/{CSV_NAME_TRAIN}.csv"
CSV_TEST = f"../../data/{TEST_COHORT}/test/{CSV_NAME_TEST}.csv"

# Image Directories
TENSOR_DIR = f"../../images/{TRAIN_COHORT}{IMG_SIZE}"
# TENSOR_DIR = f"/mnt/bulk-vega/paulkuntke/mspaths/derivatives/reg_to_mni/"
BIDS_TAGS = "space-MNI152"
TENSOR_DIR_TEST = f"../../images/{TEST_COHORT}{IMG_SIZE}"
# TENSOR_DIR_TEST = f"/mnt/bulk-vega/paulkuntke/mspaths/derivatives/reg_to_mni/"

# ============================================================================
# MODEL ARCHITECTURE SETTINGS
# ============================================================================

# Image Parameters
N_CHANNELS = 1
N_CLASSES = 2

# LoRA (Low-Rank Adaptation) Parameters
# Rank determines the dimensionality of the low-rank matrices
LORA_RANK = 16
# Alpha controls the scaling of LoRA weights (typically rank * 2 or rank * 4)
LORA_ALPHA = 64
# Which layers to apply LoRA to
LORA_TARGET_MODULES = ["feature_extractor.conv_1.0"]

# SSL (Self-Supervised Learning) Pretrained Model
SSL_COHORT = "ukb"
SSL_BATCH_SIZE = 16
SSL_EPOCHS = 1000
PRETRAINED_MODEL = (
    f"../models/ssl/sfcne/{SSL_COHORT}/{SSL_COHORT}{IMG_SIZE}/"
    f"final_model_b{SSL_BATCH_SIZE}_e{SSL_EPOCHS}.pt"
)

# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================

# Basic Training Parameters
BATCH_SIZE = 32
NUM_EPOCHS = 1000
LEARNING_RATE = 0.1
NUM_WORKERS = 8
DEVICE = args.gpu if args.gpu else "cuda"
SEED = 42
NROWS: Optional[int] = None  # Set to int for subset, None for all data

# Learning Rate Finder
USE_LR_FINDER = True

# Early Stopping
PATIENCE = 20

# Learning Rate Scheduler (ReduceLROnPlateau)
SCHEDULER_MODE = "min"  # 'min' for loss, 'max' for accuracy
SCHEDULER_FACTOR = 0.5  # Multiply LR by this factor when reducing
SCHEDULER_PATIENCE = 3  # Number of epochs with no improvement

# ============================================================================
# OUTPUT PATHS
# ============================================================================

# Experiment Naming
# Safely construct experiment name
_lora_modules_str = "_".join(LORA_TARGET_MODULES).replace(".", "-")
EXPERIMENT_NAME = (
    f"{COLUMN_NAME}_e{NUM_EPOCHS}_b{BATCH_SIZE}_im{IMG_SIZE}"
    # lora-{_lora_modules_str}"
)

# Output Directories
MODEL_DIR = "../../models"
SCORES_DIR = "../../scores"
LOG_DIR = "../../logs"
EVALUATION_DIR = "../../evaluations"
EXPLAINABILITY_DIR = "../../explainability"

# Additional Options
KAPLAN_MEIER = False

# ============================================================================
# EXPLAINABILITY & HEATMAP CONFIGURATION
# ============================================================================

# Visualization Mode
HEATMAP_MODE = "average"  # Options: 'single', 'average', 'top_individual'
HEATMAP_TOP_N = 20

# Attention Method
ATTENTION_METHOD = "saliency"  # Options: 'saliency', 'gradcam'
ATTENTION_MODE = "magnitude"  # Options: 'magnitude', 'signed'
ATTENTION_TARGET = "logit_diff"  # Options: 'logit_diff', 'pred', 'target_class'
ATTENTION_CLASS_IDX: Optional[int] = None

# Brain Atlas Configuration
ATLAS_TYPE = "AAL"  # Automated Anatomical Labeling
ATLAS_PATH = "utils/aal3_resampled_96.nii.gz"
N_REGIONS = 40  # Number of top regions to analyze

# ============================================================================
# SSL (SELF-SUPERVISED LEARNING) CONFIGURATION
# ============================================================================

# Basic SSL Parameters
SEED_SSL = 42
COHORT_SSL = "ukb"
MAX_EPOCHS_SSL = 1000
VAL_INTERVAL_SSL = 1
BATCH_SIZE_SSL = 16
LEARNING_RATE_SSL = 1e-1
NUM_WORKERS_SSL = 8
MODEL_TYPE_SSL = "sfcne"
MODEL_NAME_SSL = f"final_model_b{BATCH_SIZE_SSL}_e{MAX_EPOCHS_SSL}.pt"

# SSL Loss Parameters
CONTRASTIVE_TEMPERATURE = 0.05  # Temperature for contrastive loss

# SSL Training Control
PATIENCE_SSL = 10
SCHEDULER_FACTOR_SSL = 0.5
SCHEDULER_PATIENCE_SSL = 5

# SSL Paths
JSON_PATH = f"../data/ssl_data/ssl-{COHORT_SSL}.json"
LOG_DIR_SSL = f"../logs/ssl/{MODEL_TYPE_SSL}/{COHORT_SSL}/{COHORT_SSL}{IMG_SIZE}/"
MODEL_DIR_SSL = f"../models/ssl/{MODEL_TYPE_SSL}/{COHORT_SSL}/{COHORT_SSL}{IMG_SIZE}/"

# ============================================================================
# DIMENSIONALITY REDUCTION SETTINGS (for visualization)
# ============================================================================

# Method Selection
REDUCTION_METHOD = "umap"  # Options: 'umap', 'tsne', 'pca'

# UMAP Parameters
N_NEIGHBORS = 6  # Number of neighbors (controls local vs global structure)
MIN_DIST = 1  # Minimum distance between points in embedding

# t-SNE Parameters
PERPLEXITY = 100  # Balance between local and global structure (typically 5-50)

# General Parameters
RANDOM_STATE = 42

# ============================================================================
# FEATURE EXTRACTION & VISUALIZATION SETTINGS
# ============================================================================

# Cohort for Feature Extraction
COHORT_EXTRACT = "BrainLat"
CSV_NAME_EXTRACT = "ms-cn_balanced"
DATA_PATH = f"../../data/{COHORT_EXTRACT}/{CSV_NAME_EXTRACT}.csv"

# Visualization Parameters
POINT_SIZE = 600
TRANSPARENCY = 0.6
FONTSIZE_MAX = 40
FONTSIZE_MIN = 30

# Color Mapping for Diagnostic Categories
DIAGNOSIS_COLORS: Dict[str, str] = {
    "CN": "lightgrey",  # Controls
    "AD": "#FFA500",  # Alzheimer's Disease - Orange
    "FTD": "#6495ED",  # Frontotemporal Dementia - Cornflower blue
    "PD": "#2E8B57",  # Parkinson's Disease - Sea green
    "MS": "#F48FB1",  # Multiple Sclerosis - Light pink
    "BV": "#9370DB",  # Behavioral variant FTD - Medium purple
    "SV": "#20B2AA",  # Semantic variant PPA - Light sea green
    "PNFA": "#FF6347",  # Non-fluent variant PPA - Tomato red
}

# Feature Extraction Paths
IMAGES_EXT_DIR = f"../../images/{COHORT_EXTRACT}/"
FEATURES_EXT_DIR = f"../../features/{COHORT_EXTRACT}/{MODEL_TYPE_SSL}/"
VIZ_DIR = (
    f"../../representations/{COHORT_EXTRACT}/{MODEL_TYPE_SSL}/"
    f"ssl-{COHORT_SSL}/{MODEL_NAME_SSL}/{REDUCTION_METHOD}/"
)
DATA_PATH_EXTRACT = f"../../data/{COHORT_EXTRACT}/{CSV_NAME_EXTRACT}.csv"

# ============================================================================
# PREPROCESSING SETTINGS
# ============================================================================

# Cohort Configuration
PREPROCESS_COHORT = "trial"
REGISTRATION_TYPE = "Affine"  # Registration algorithm type
CROP_SIZE = 180  # Size before downsampling
PREPROCESS_IMG_SIZE = 180  # Final image size

# Template and Tools
TEMPLATE_PATH = (
    "../images/templates/mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c.nii"
)
DCM2NIIX = "../.venv/bin/dcm2niix"

# Processing Directories
DCM_FOLDER = f"../images/{PREPROCESS_COHORT}/dcm_raw/"
INPUT_FOLDER = f"../images/{PREPROCESS_COHORT}/nifti_raw/"
N4_FOLDER = f"../images/{PREPROCESS_COHORT}/nifti_n4/"
REG_FOLDER = f"../images/{PREPROCESS_COHORT}/nifti_reg_{REGISTRATION_TYPE}/"
DESKULL_FOLDER = f"../images/{PREPROCESS_COHORT}/nifti_deskull_{REGISTRATION_TYPE}/"
NPY_FOLDER = f"../images/{PREPROCESS_COHORT}/npy{PREPROCESS_IMG_SIZE}/"

# Processing Parameters
N4_PROCESSES = 4  # Number of parallel processes for N4 bias correction
REG_PROCESSES = 4  # Number of parallel processes for registration
GPU_ID = 0  # GPU device ID for processing

# ============================================================================
# CONFIGURATION VALIDATION
# ============================================================================


def validate_config() -> List[str]:
    """
    Validate configuration settings and return list of warnings.

    Returns:
        List of warning messages (empty if all valid)
    """
    warnings = []

    # Check critical paths exist
    critical_paths = {
        "Template": TEMPLATE_PATH,
        "DCM2NIIX": DCM2NIIX,
    }

    for name, path in critical_paths.items():
        if path and not os.path.exists(path):
            warnings.append(f"{name} path does not exist: {path}")

    # Validate hyperparameters
    if BATCH_SIZE <= 0:
        warnings.append(f"BATCH_SIZE must be positive, got {BATCH_SIZE}")

    if LEARNING_RATE <= 0 or LEARNING_RATE >= 1:
        warnings.append(f"LEARNING_RATE should be in (0, 1), got {LEARNING_RATE}")

    if N_CLASSES < 2:
        warnings.append(f"N_CLASSES must be >= 2, got {N_CLASSES}")

    # Validate device
    if not DEVICE.startswith("cuda") and DEVICE != "cpu":
        warnings.append(f"Invalid DEVICE: {DEVICE}. Should be 'cpu' or 'cuda:X'")

    # Check mode-specific settings
    if TRAINING_MODE == "lora":
        if not LORA_TARGET_MODULES:
            warnings.append("LORA_TARGET_MODULES is empty")
        if LORA_RANK <= 0:
            warnings.append(f"LORA_RANK must be positive, got {LORA_RANK}")

    # Validate attention settings
    valid_attention_methods = ["saliency", "gradcam"]
    if ATTENTION_METHOD not in valid_attention_methods:
        warnings.append(f"Invalid ATTENTION_METHOD: {ATTENTION_METHOD}")

    valid_attention_modes = ["magnitude", "signed"]
    if ATTENTION_MODE not in valid_attention_modes:
        warnings.append(f"Invalid ATTENTION_MODE: {ATTENTION_MODE}")

    return warnings


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def print_config_summary():
    """Print a summary of key configuration settings."""
    print("=" * 70)
    print("CONFIGURATION SUMMARY")
    print("=" * 70)
    print(f"Training Mode: {TRAINING_MODE}")
    print(f"Target Column: {COLUMN_NAME}")
    print(f"Train Cohort: {TRAIN_COHORT}")
    print(f"Test Cohort: {TEST_COHORT}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Device: {DEVICE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Experiment: {EXPERIMENT_NAME}")
    print("=" * 70)


def create_output_directories():
    """Create all necessary output directories."""
    dirs_to_create = [
        MODEL_DIR,
        SCORES_DIR,
        LOG_DIR,
        EVALUATION_DIR,
        EXPLAINABILITY_DIR,
        LOG_DIR_SSL,
        MODEL_DIR_SSL,
        FEATURES_EXT_DIR,
        VIZ_DIR,
    ]

    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)
