"""
SVM Baseline Classifier for thermal time-series classification.

Uses Support Vector Machine with RBF kernel on 5 statistical features.
Grid search is used to find optimal hyperparameters (C, gamma).

This serves as the baseline for comparison with deep learning methods.
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib
from pathlib import Path
from typing import Dict, Tuple, Optional
import json
from datetime import datetime

import sys
CODE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(CODE_DIR))

from thermal_classifiers.shared.config import (
    CLASS_NAMES, RANDOM_SEED, RESULTS_DIR
)
from thermal_classifiers.shared.evaluation import (
    compute_metrics, compute_confusion_matrix, plot_confusion_matrix, bootstrap_ci
)


# Grid search parameter space
PARAM_GRID = {
    'svm__C': [0.1, 1.0, 10.0, 100.0],
    'svm__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0],
    'svm__kernel': ['rbf', 'poly', 'sigmoid']
}

# Reduced grid for faster search
PARAM_GRID_FAST = {
    'svm__C': [0.1, 1.0, 10.0],
    'svm__gamma': ['scale', 0.01, 0.1],
    'svm__kernel': ['rbf']
}


def create_svm_pipeline(
    C: float = 1.0,
    gamma: str = 'scale',
    kernel: str = 'rbf',
    class_weight: str = 'balanced'
) -> Pipeline:
    """
    Create an SVM pipeline with StandardScaler.

    Args:
        C: Regularization parameter
        gamma: Kernel coefficient
        kernel: Kernel type ('rbf', 'poly', 'sigmoid')
        class_weight: 'balanced' for handling class imbalance

    Returns:
        sklearn Pipeline
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(
            C=C,
            gamma=gamma,
            kernel=kernel,
            class_weight=class_weight,
            random_state=RANDOM_SEED,
            probability=True  # Enable probability estimates for calibration
        ))
    ])


def grid_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv: int = 5,
    fast: bool = False,
    verbose: int = 1
) -> Tuple[Pipeline, Dict]:
    """
    Perform grid search to find optimal SVM hyperparameters.

    Args:
        X_train: Training features
        y_train: Training labels
        cv: Number of cross-validation folds
        fast: Use reduced parameter grid
        verbose: Verbosity level

    Returns:
        Tuple of (best_pipeline, cv_results_dict)
    """
    param_grid = PARAM_GRID_FAST if fast else PARAM_GRID

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(class_weight='balanced', random_state=RANDOM_SEED, probability=True))
    ])

    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=verbose,
        refit=True
    )

    grid.fit(X_train, y_train)

    results = {
        'best_params': grid.best_params_,
        'best_score': grid.best_score_,
        'cv_results': {
            'mean_test_score': grid.cv_results_['mean_test_score'].tolist(),
            'std_test_score': grid.cv_results_['std_test_score'].tolist(),
            'params': [str(p) for p in grid.cv_results_['params']]
        }
    }

    return grid.best_estimator_, results


def train_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    C: float = 1.0,
    gamma: str = 'scale',
    kernel: str = 'rbf'
) -> Tuple[Pipeline, Dict]:
    """
    Train SVM with specified hyperparameters.

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        C: Regularization parameter
        gamma: Kernel coefficient
        kernel: Kernel type

    Returns:
        Tuple of (trained_pipeline, metrics_dict)
    """
    pipeline = create_svm_pipeline(C=C, gamma=gamma, kernel=kernel)
    pipeline.fit(X_train, y_train)

    # Evaluate on validation set
    y_pred = pipeline.predict(X_val)
    metrics = compute_metrics(y_val, y_pred)

    return pipeline, metrics


def evaluate_svm(
    model: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    split_name: str = 'test'
) -> Dict:
    """
    Evaluate trained SVM model.

    Args:
        model: Trained SVM pipeline
        X: Features
        y: True labels
        split_name: Name of the data split (for logging)

    Returns:
        Dict with metrics, predictions, and probabilities
    """
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)

    metrics = compute_metrics(y, y_pred)

    # Bootstrap confidence interval for accuracy
    point, lower, upper = bootstrap_ci(y, y_pred, lambda yt, yp: np.mean(yt == yp))

    results = {
        'metrics': metrics,
        'predictions': y_pred,
        'probabilities': y_prob,
        'accuracy_ci': {'point': point, 'lower': lower, 'upper': upper},
        'split': split_name
    }

    return results


def save_model(
    model: Pipeline,
    output_dir: Path,
    model_name: str = 'svm_baseline'
):
    """Save trained model and scaler."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / f'{model_name}.joblib'
    joblib.dump(model, model_path)

    return model_path


def load_model(model_path: Path) -> Pipeline:
    """Load trained model."""
    return joblib.load(model_path)


def save_results(
    results: Dict,
    output_dir: Path,
    model_name: str = 'svm_baseline'
):
    """Save evaluation results to JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert numpy arrays to lists for JSON serialization
    serializable = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            serializable[key] = value.tolist()
        elif isinstance(value, dict):
            serializable[key] = {}
            for k, v in value.items():
                if isinstance(v, np.ndarray):
                    serializable[key][k] = v.tolist()
                elif isinstance(v, (np.floating, float)):
                    serializable[key][k] = float(v)
                elif isinstance(v, (np.integer, int)):
                    serializable[key][k] = int(v)
                else:
                    serializable[key][k] = v
        elif isinstance(value, (np.floating, float)):
            serializable[key] = float(value)
        else:
            serializable[key] = value

    serializable['timestamp'] = datetime.now().isoformat()

    with open(output_dir / f'{model_name}_results.json', 'w') as f:
        json.dump(serializable, f, indent=2)


if __name__ == '__main__':
    from feature_extraction import load_and_extract_features, TRAIN_DATA_PATH, VAL_DATA_PATH, TEST_DATA_PATH

    print("Testing SVM classifier...")

    # Load data
    X_train, y_train, _ = load_and_extract_features(str(TRAIN_DATA_PATH))
    X_val, y_val, _ = load_and_extract_features(str(VAL_DATA_PATH))
    X_test, y_test, ids_test = load_and_extract_features(str(TEST_DATA_PATH))

    print(f"\nTrain: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Grid search
    print("\nPerforming grid search (fast mode)...")
    best_model, grid_results = grid_search(X_train, y_train, cv=3, fast=True)
    print(f"Best params: {grid_results['best_params']}")
    print(f"Best CV score: {grid_results['best_score']:.4f}")

    # Evaluate on validation
    val_results = evaluate_svm(best_model, X_val, y_val, 'validation')
    print(f"\nValidation results:")
    print(f"  Accuracy: {val_results['metrics']['accuracy']:.4f}")
    print(f"  F1 Weighted: {val_results['metrics']['f1_weighted']:.4f}")

    # Evaluate on test
    test_results = evaluate_svm(best_model, X_test, y_test, 'test')
    print(f"\nTest results:")
    print(f"  Accuracy: {test_results['metrics']['accuracy']:.4f}")
    print(f"  F1 Weighted: {test_results['metrics']['f1_weighted']:.4f}")
    print(f"  95% CI: [{test_results['accuracy_ci']['lower']:.4f}, {test_results['accuracy_ci']['upper']:.4f}]")

    # Classification report
    print("\nClassification Report (Test):")
    print(classification_report(y_test, test_results['predictions'], target_names=CLASS_NAMES))

    print("\nSVM classifier test passed!")
