#!/usr/bin/env python3
"""
MiniRocket classifier for thermal time-series classification.

MiniRocket (Minimal RandOm Convolutional KErnel Transform) generates random
convolutional kernels and applies them to the input sequence to create
high-dimensional features, which are then classified with logistic regression.

Key advantages for small datasets:
- Random kernels = no learnable parameters = cannot overfit
- Ridge regression classifier is robust on small data
- SOTA on UCR datasets with <100 samples
- No hyperparameter tuning needed

Usage:
    python -m classifiers.thermal_classifiers.minirocket.train
    python -m classifiers.thermal_classifiers.minirocket.train --num-kernels 10000
    python -m classifiers.thermal_classifiers.minirocket.train --output-dir custom_dir
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import json
import numpy as np
import joblib

CODE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(CODE_DIR))

from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import StratifiedKFold
from thermal_classifiers.minirocket.minirocket_multivariate import MiniRocketMultivariate
from thermal_classifiers.shared.config import (
    RESULTS_DIR, CLASS_NAMES, RANDOM_SEED, set_seed, MAX_SEQ_LEN,
    TRAIN_DATA_PATH, VAL_DATA_PATH, TEST_DATA_PATH, print_config
)
from thermal_classifiers.shared.dataset import load_data
from thermal_classifiers.shared.evaluation import (
    compute_metrics, compute_confusion_matrix, plot_confusion_matrix,
    save_predictions, save_classification_report
)


def subsample_sequences(sequences, target_len=MAX_SEQ_LEN):
    """
    Subsample sequences to uniform length, preserving full temporal dynamics.

    Instead of truncating (which loses late-sequence information),
    we uniformly subsample to capture the entire heating/cooling cycle.

    MiniRocket requires 3D input: [n_samples, n_channels, seq_len]
    For univariate thermal data: [n_samples, 1, seq_len]

    Args:
        sequences: List of 1D numpy arrays (variable length)
        target_len: Target sequence length

    Returns:
        X: numpy array of shape [n_samples, 1, target_len]
    """
    n_samples = len(sequences)
    X = np.zeros((n_samples, target_len))

    for i, seq in enumerate(sequences):
        seq = np.array(seq)
        orig_len = len(seq)

        if orig_len > target_len:
            # Subsample uniformly to preserve full temporal structure
            indices = np.linspace(0, orig_len - 1, target_len, dtype=int)
            X[i] = seq[indices]
        elif orig_len < target_len:
            # Pad with last value (constant padding preserves thermal level)
            X[i, :orig_len] = seq
            X[i, orig_len:] = seq[-1]  # Repeat last value instead of zeros
        else:
            X[i] = seq

    # Reshape to [n_samples, 1, seq_len] for univariate MiniRocket
    return X.reshape(n_samples, 1, target_len)


def cross_validate(X, y, num_kernels=10000, n_folds=5):
    """
    Run stratified cross-validation.

    Args:
        X: Input data (n_samples, n_channels, n_timepoints)
        y: Labels (n_samples,)
        num_kernels: Number of MiniRocket kernels
        n_folds: Number of CV folds

    Returns:
        results: Dictionary with CV scores
    """
    from sklearn.metrics import f1_score, accuracy_score

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)

    f1_macro_scores = []
    accuracy_scores_list = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold + 1}/{n_folds} ---")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Fit MiniRocket
        rocket = MiniRocketMultivariate(num_kernels=num_kernels, random_state=RANDOM_SEED + fold)
        X_train_transform = rocket.fit_transform(X_train)
        X_val_transform = rocket.transform(X_val)

        # Train Ridge Classifier
        classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
        classifier.fit(X_train_transform, y_train)

        # Predict
        y_pred = classifier.predict(X_val_transform)

        # Metrics
        f1_macro = f1_score(y_val, y_pred, average='macro')
        accuracy = accuracy_score(y_val, y_pred)

        f1_macro_scores.append(f1_macro)
        accuracy_scores_list.append(accuracy)

        print(f"  F1 Macro: {f1_macro:.4f}, Accuracy: {accuracy:.4f}")

    results = {
        'f1_macro_mean': float(np.mean(f1_macro_scores)),
        'f1_macro_std': float(np.std(f1_macro_scores)),
        'accuracy_mean': float(np.mean(accuracy_scores_list)),
        'accuracy_std': float(np.std(accuracy_scores_list)),
        'fold_scores': {
            'f1_macro': [float(x) for x in f1_macro_scores],
            'accuracy': [float(x) for x in accuracy_scores_list],
        }
    }

    print(f"\nCV Results:")
    print(f"  F1 Macro: {results['f1_macro_mean']:.4f} ± {results['f1_macro_std']:.4f}")
    print(f"  Accuracy: {results['accuracy_mean']:.4f} ± {results['accuracy_std']:.4f}")

    return results


def main(num_kernels=10000, output_dir=None):
    """
    Main training pipeline for MiniRocket classifier.

    Args:
        num_kernels: Number of random convolutional kernels (default: 10000)
        output_dir: Directory to save results (default: RESULTS_DIR / 'minirocket')

    Returns:
        0 on success, 1 on failure
    """
    print("=" * 70)
    print("MiniRocket THERMAL CLASSIFIER TRAINING")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Num Kernels: {num_kernels}")
    print()

    # Set seed for reproducibility
    set_seed(RANDOM_SEED)

    # Setup output directory
    if output_dir is None:
        output_dir = RESULTS_DIR / 'minirocket'
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output Directory: {output_dir}")
    print()

    # ==========================================================================
    # STEP 1: Load Data
    # ==========================================================================
    print("=" * 70)
    print("STEP 1: Loading Data")
    print("=" * 70)

    train_seqs, y_train, ids_train = load_data(str(TRAIN_DATA_PATH))
    val_seqs, y_val, ids_val = load_data(str(VAL_DATA_PATH))
    test_seqs, y_test, ids_test = load_data(str(TEST_DATA_PATH))

    # FAIRNESS FIX: Train on train only (not train+val) for fair comparison
    print(f"Train samples: {len(train_seqs)} (val unused for MiniRocket - no early stopping)")
    print(f"Test samples: {len(test_seqs)}")

    # Class distribution
    unique, counts = np.unique(y_train, return_counts=True)
    print("\nTraining class distribution:")
    for cls_idx, count in zip(unique, counts):
        print(f"  {CLASS_NAMES[cls_idx]}: {count}")
    print()

    # ==========================================================================
    # STEP 2: Preprocess Sequences
    # ==========================================================================
    print("=" * 70)
    print("STEP 2: Preprocessing Sequences")
    print("=" * 70)

    # Subsample to uniform length (preserves full temporal dynamics)
    X_train = subsample_sequences(train_seqs, MAX_SEQ_LEN)
    X_test = subsample_sequences(test_seqs, MAX_SEQ_LEN)

    y_train_arr = np.array(y_train)
    y_test_arr = np.array(y_test)

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print()

    # ==========================================================================
    # STEP 3: Cross-Validation
    # ==========================================================================
    print("=" * 70)
    print("STEP 3: Cross-Validation")
    print("=" * 70)

    cv_results = cross_validate(X_train, y_train_arr, num_kernels=num_kernels, n_folds=5)

    # ==========================================================================
    # STEP 4: Final Training and Test Evaluation
    # ==========================================================================
    print()
    print("=" * 70)
    print("STEP 4: Final Training and Test Evaluation")
    print("=" * 70)

    print("\n=== Training Final Model ===")
    print(f"Training with {num_kernels} random kernels...")
    start_time = datetime.now()

    # Fit MiniRocket transform
    rocket = MiniRocketMultivariate(num_kernels=num_kernels, random_state=RANDOM_SEED)
    X_train_transform = rocket.fit_transform(X_train)
    X_test_transform = rocket.transform(X_test)

    print(f"Transformed shape: {X_train_transform.shape}")

    # Train Ridge Classifier
    classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
    classifier.fit(X_train_transform, y_train_arr)

    train_time = (datetime.now() - start_time).total_seconds()
    print(f"Best alpha: {classifier.alpha_:.6f}")
    print(f"Training completed in {train_time:.2f} seconds")

    # Predict
    print("\n=== Test Results ===")
    y_pred = classifier.predict(X_test_transform)

    # Get probabilities via softmax of decision values
    decision = classifier.decision_function(X_test_transform)
    probs = np.exp(decision) / np.exp(decision).sum(axis=1, keepdims=True)

    # Compute metrics
    metrics = compute_metrics(y_test_arr, y_pred)
    metrics['num_parameters'] = num_kernels  # Feature count (kernels)
    metrics['training_time_seconds'] = train_time

    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test F1 (Weighted): {metrics['f1_weighted']:.4f}")
    print(f"Test F1 (Macro): {metrics['f1_macro']:.4f}")
    print()

    # Per-class metrics
    print("Per-Class Metrics:")
    for cls_name in CLASS_NAMES:
        if cls_name in metrics['per_class']:
            cls_metrics = metrics['per_class'][cls_name]
            print(f"  {cls_name}: P={cls_metrics['precision']:.3f}, "
                  f"R={cls_metrics['recall']:.3f}, F1={cls_metrics['f1']:.3f}")
    print()

    # ==========================================================================
    # STEP 5: Save Results
    # ==========================================================================
    print("=" * 70)
    print("STEP 5: Saving Results")
    print("=" * 70)

    # Merge CV results into metrics
    all_metrics = {**metrics, **cv_results}
    all_metrics['ridge_alpha'] = float(classifier.alpha_)

    # Test metrics (with CV results)
    with open(output_dir / 'test_metrics.json', 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Saved: test_metrics.json")

    # Predictions
    save_predictions(
        y_test_arr, y_pred, ids_test,
        output_dir / 'test_predictions.csv',
        probabilities=probs
    )
    print(f"Saved: test_predictions.csv")

    # Classification report
    save_classification_report(
        y_test_arr, y_pred,
        output_dir / 'classification_report.txt'
    )
    print(f"Saved: classification_report.txt")

    # Confusion matrix
    cm = compute_confusion_matrix(y_test_arr, y_pred)
    np.save(output_dir / 'confusion_matrix.npy', cm)
    plot_confusion_matrix(
        cm, CLASS_NAMES,
        title='MiniRocket Confusion Matrix',
        save_path=output_dir / 'confusion_matrix.png'
    )
    print(f"Saved: confusion_matrix.npy, confusion_matrix.png")

    # Model (save both transformer and classifier)
    model_dict = {
        'rocket': rocket,
        'classifier': classifier,
    }
    joblib.dump(model_dict, output_dir / 'model.joblib')
    print(f"Saved: model.joblib")

    # Best params
    best_params = {
        'model_type': 'minirocket',
        'best_params': {
            'num_kernels': num_kernels,
            'ridge_alpha': float(classifier.alpha_),
        },
        'timestamp': datetime.now().isoformat()
    }
    with open(output_dir / 'best_params.json', 'w') as f:
        json.dump(best_params, f, indent=2)
    print(f"Saved: best_params.json")

    # ==========================================================================
    # Summary
    # ==========================================================================
    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"CV F1 Macro:   {cv_results['f1_macro_mean']:.4f} ± {cv_results['f1_macro_std']:.4f}")
    print(f"CV Accuracy:   {cv_results['accuracy_mean']:.4f} ± {cv_results['accuracy_std']:.4f}")
    print(f"Test F1 Macro: {metrics['f1_macro']:.4f}")
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Training Time: {train_time:.2f}s")
    print(f"All outputs saved to: {output_dir}")
    print("=" * 70)

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train MiniRocket classifier for thermal time-series'
    )
    parser.add_argument(
        '--num-kernels', type=int, default=10000,
        help='Number of random convolutional kernels (default: 10000)'
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Output directory for results'
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None
    sys.exit(main(num_kernels=args.num_kernels, output_dir=output_dir))
