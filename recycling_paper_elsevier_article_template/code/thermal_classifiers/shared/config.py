"""
Self-Contained Configuration for Thermal Classification.

All parameters are inlined here for reproducibility.
Values match the RGB config used for fair cross-domain comparison.
"""

import os
import random
from pathlib import Path

# =============================================================================
# RANDOM SEED (Reproducibility)
# =============================================================================
RANDOM_SEED = 42

# =============================================================================
# CLASS DEFINITIONS
# =============================================================================
NUM_CLASSES = 4
CLASS_NAMES = ['metal', 'plastic', 'glass', 'paper']  # STANDARDIZED ORDER
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}

# =============================================================================
# DATA PATHS
# =============================================================================
CODE_DIR = Path(__file__).parent.parent.parent  # code/
TRAIN_DATA_PATH = CODE_DIR / 'data' / 'final_thermal_train_dataset.csv'
VAL_DATA_PATH = CODE_DIR / 'data' / 'final_thermal_validation_dataset.csv'
TEST_DATA_PATH = CODE_DIR / 'data' / 'final_thermal_test_dataset.csv'

# Output directories
THERMAL_CLASSIFIERS_DIR = Path(__file__).parent.parent  # thermal_classifiers/


def _get_results_dir() -> Path:
    """Get results directory, respecting EXPERIMENT_OUTPUT_DIR override."""
    override = os.environ.get('EXPERIMENT_OUTPUT_DIR')
    if override:
        return Path(override)
    return THERMAL_CLASSIFIERS_DIR / 'results'


RESULTS_DIR = _get_results_dir()
BEST_PARAMS_DIR = Path(__file__).parent / 'best_hyperparameters'

# =============================================================================
# SEQUENCE PROCESSING (aligned with RGB for fair comparison)
# =============================================================================
MAX_SEQ_LEN = 2000
MIN_SEQ_LEN = 10
SUBSAMPLE_STRIDE = 2

# =============================================================================
# THERMAL-SPECIFIC: INPUT SIZE
# =============================================================================
INPUT_SIZE = 1  # Univariate thermal intensity (RGB uses 4 probability distributions)

# =============================================================================
# TRAINING PARAMETERS (aligned with RGB for fair comparison)
# =============================================================================
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
EARLY_STOPPING_MIN_DELTA = 0.001
BATCH_SIZE = 16
WEIGHT_DECAY = 0.1
GRADIENT_CLIP_NORM = 1.0
LEARNING_RATE_DEFAULT = 1e-3

# =============================================================================
# OPTUNA HYPERPARAMETER TUNING (aligned with RGB for fair comparison)
# =============================================================================
OPTUNA_N_TRIALS = 20
OPTUNA_CV_FOLDS = 5
OPTUNA_TUNING_EPOCHS = 30
OPTUNA_TIMEOUT = None
OPTUNA_SAMPLER = 'TPE'
OPTUNA_PRUNER = 'MedianPruner'

# =============================================================================
# STATISTICAL TESTING
# =============================================================================
NUM_METHODS = 9
NUM_COMPARISONS = NUM_METHODS * (NUM_METHODS - 1) // 2  # = 36
SIGNIFICANCE_LEVEL = 0.05
BONFERRONI_ALPHA = SIGNIFICANCE_LEVEL / NUM_COMPARISONS  # = 0.00139

# =============================================================================
# GPU/CUDA SETTINGS
# =============================================================================
USE_CUDA = True
CUDA_DETERMINISTIC = True

# =============================================================================
# LOGGING AND OUTPUT
# =============================================================================
LOG_INTERVAL = 10
SAVE_INTERVAL = 5
VERBOSE = True

# Output file names
PREDICTIONS_FILENAME = 'test_predictions.csv'
CONFUSION_MATRIX_FILENAME = 'confusion_matrix.png'
CLASSIFICATION_REPORT_FILENAME = 'classification_report.txt'
BEST_PARAMS_FILENAME = 'best_params.json'
TRAINING_HISTORY_FILENAME = 'training_history.json'
METRICS_FILENAME = 'test_metrics.json'

# =============================================================================
# SVM BASELINE FEATURES
# =============================================================================
SVM_FEATURES = ['mean', 'median', 'std', 'rise_time', 'fall_time']
RISE_FALL_THRESHOLD = 0.7

# =============================================================================
# DATA AUGMENTATION (Physics-Informed)
# =============================================================================
AUG_TIME_MASK_PROB = 0.15
AUG_TIME_MASK_MAX_FRAC = 0.08
AUG_NOISE_PROB = 1.0
AUG_NOISE_STD = 0.05
AUG_BASELINE_SHIFT_PROB = 0.40
AUG_BASELINE_SHIFT_RANGE = 0.2
AUG_INTENSITY_SCALE_PROB = 0.30
AUG_INTENSITY_SCALE_MIN = 0.92
AUG_INTENSITY_SCALE_MAX = 1.08
AUG_TIME_STRETCH_PROB = 0.25
AUG_TIME_STRETCH_MIN = 0.9
AUG_TIME_STRETCH_MAX = 1.1
USE_LEGACY_AUGMENTATION = True


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def get_device():
    """Get the appropriate device for training."""
    import torch
    if USE_CUDA and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def set_seed(seed: int = RANDOM_SEED):
    """Set random seed for reproducibility across all libraries."""
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if CUDA_DETERMINISTIC:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            try:
                torch.use_deterministic_algorithms(True)
                os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            except Exception as e:
                print(f"Warning: Could not enable full deterministic algorithms: {e}")
    except ImportError:
        pass


def validate_paths():
    """Validate that all thermal data paths exist."""
    missing = []
    for name, path in [
        ('TRAIN_DATA_PATH', TRAIN_DATA_PATH),
        ('VAL_DATA_PATH', VAL_DATA_PATH),
        ('TEST_DATA_PATH', TEST_DATA_PATH),
    ]:
        if not path.exists():
            missing.append(f"{name}: {path}")

    if missing:
        raise FileNotFoundError(
            f"Missing data files:\n" + "\n".join(missing)
        )


def print_config():
    """Print all configuration values for logging/reproducibility."""
    print("=" * 70)
    print("THERMAL CLASSIFICATION - CONFIGURATION")
    print("=" * 70)
    print()
    print("TRAINING PARAMETERS:")
    print(f"  Random Seed: {RANDOM_SEED}")
    print(f"  Num Classes: {NUM_CLASSES}")
    print(f"  Class Names: {CLASS_NAMES}")
    print(f"  Max Sequence Length: {MAX_SEQ_LEN}")
    print(f"  Max Epochs: {MAX_EPOCHS}")
    print(f"  Early Stopping Patience: {EARLY_STOPPING_PATIENCE}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Weight Decay: {WEIGHT_DECAY}")
    print(f"  Gradient Clip Norm: {GRADIENT_CLIP_NORM}")
    print(f"  Optuna Trials: {OPTUNA_N_TRIALS}")
    print(f"  Optuna CV Folds: {OPTUNA_CV_FOLDS}")
    print(f"  Optuna Tuning Epochs: {OPTUNA_TUNING_EPOCHS}")
    print()
    print("THERMAL-SPECIFIC:")
    print(f"  Input Size: {INPUT_SIZE} (univariate thermal intensity)")
    print(f"  Train Data: {TRAIN_DATA_PATH}")
    print(f"  Val Data:   {VAL_DATA_PATH}")
    print(f"  Test Data:  {TEST_DATA_PATH}")
    print(f"  Results:    {RESULTS_DIR}")
    print()
    print("DATA AUGMENTATION:")
    print(f"  Use Legacy: {USE_LEGACY_AUGMENTATION}")
    print()
    print(f"Device: {get_device()}")
    print("=" * 70)


if __name__ == '__main__':
    print_config()
    try:
        validate_paths()
        print("\nAll data paths validated successfully!")
    except FileNotFoundError as e:
        print(f"\nWARNING: {e}")
