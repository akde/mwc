#!/usr/bin/env python3
"""
Training script for SVM baseline classifier.

This script:
1. Loads thermal time-series data
2. Extracts 5 statistical features (mean, median, std, rise_time, fall_time)
3. Performs grid search for optimal hyperparameters
4. Trains final model and evaluates on test set
5. Saves model, results, and visualizations

Usage:
    python train.py [--fast] [--output-dir OUTPUT_DIR]
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import json
import numpy as np

CODE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(CODE_DIR))

from thermal_classifiers.shared.feature_extraction import (
    load_and_extract_features, get_feature_dataframe, SVM_FEATURES
)
from thermal_classifiers.svm.model import (
    grid_search, evaluate_svm, save_model, save_results
)
from thermal_classifiers.shared.config import (
    TRAIN_DATA_PATH, VAL_DATA_PATH, TEST_DATA_PATH, RESULTS_DIR,
    CLASS_NAMES, RANDOM_SEED, OPTUNA_N_TRIALS, print_config
)
from thermal_classifiers.shared.evaluation import (
    compute_confusion_matrix, plot_confusion_matrix, plot_learning_curves,
    per_class_to_latex, save_predictions, save_classification_report
)


def main(
    skip_tuning: bool = False,
    n_trials: int = OPTUNA_N_TRIALS,
    output_dir: Path = None
):
    """
    Main training pipeline for SVM.

    Args:
        skip_tuning: If True, use reduced grid search (fast mode)
        n_trials: Number of trials (currently unused - SVM uses grid search)
        output_dir: Output directory for results
    """
    print("=" * 60)
    print("SVM BASELINE TRAINING")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    # Set up output directory
    if output_dir is None:
        output_dir = RESULTS_DIR / 'svm'
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print(f"Mode: {'fast (reduced grid)' if skip_tuning else 'full grid search'}")
    if not skip_tuning and n_trials != OPTUNA_N_TRIALS:
        print(f"Note: --n-trials={n_trials} ignored (SVM uses grid search, not Optuna)")
    print()

    # Print configuration
    print_config()

    # =========================================================================
    # 1. Load and extract features
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 1: Load data and extract features")
    print("=" * 60)

    X_train, y_train, ids_train = load_and_extract_features(str(TRAIN_DATA_PATH))
    X_val, y_val, ids_val = load_and_extract_features(str(VAL_DATA_PATH))
    X_test, y_test, ids_test = load_and_extract_features(str(TEST_DATA_PATH))

    print(f"\nData shapes:")
    print(f"  Train: {X_train.shape} ({len(X_train)} samples)")
    print(f"  Val:   {X_val.shape} ({len(X_val)} samples)")
    print(f"  Test:  {X_test.shape} ({len(X_test)} samples)")

    # Save feature statistics
    train_df = get_feature_dataframe(str(TRAIN_DATA_PATH))
    feature_stats = train_df.groupby('class')[SVM_FEATURES].agg(['mean', 'std'])
    feature_stats.to_csv(output_dir / 'feature_statistics.csv')
    print(f"\nFeature statistics saved to: {output_dir / 'feature_statistics.csv'}")

    # =========================================================================
    # 2. Grid search for hyperparameters
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 2: Grid search for optimal hyperparameters")
    print("=" * 60)

    # Combine train and validation for grid search CV
    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])

    print(f"Grid search on {len(X_trainval)} samples...")

    best_model, grid_results = grid_search(
        X_trainval, y_trainval,
        cv=5,
        fast=skip_tuning,  # skip_tuning = fast mode (reduced grid)
        verbose=2
    )

    print(f"\nBest parameters found:")
    for param, value in grid_results['best_params'].items():
        print(f"  {param}: {value}")
    print(f"Best CV weighted F1: {grid_results['best_score']:.4f}")

    # Save grid search results
    with open(output_dir / 'grid_search_results.json', 'w') as f:
        json.dump(grid_results, f, indent=2)

    # =========================================================================
    # 3. Train final model and evaluate
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 3: Train final model and evaluate")
    print("=" * 60)

    # The best model is already fitted from grid search
    # Evaluate on train set (to check for overfitting)
    train_results = evaluate_svm(best_model, X_trainval, y_trainval, 'train')
    print(f"\nTraining set results:")
    print(f"  Accuracy:    {train_results['metrics']['accuracy']*100:.2f}%")
    print(f"  Weighted F1: {train_results['metrics']['f1_weighted']:.4f}")
    print(f"  Macro F1:    {train_results['metrics']['f1_macro']:.4f}")

    # Evaluate on test set
    test_results = evaluate_svm(best_model, X_test, y_test, 'test')
    print(f"\nTest set results:")
    print(f"  Accuracy:    {test_results['metrics']['accuracy']*100:.2f}%")
    print(f"  Weighted F1: {test_results['metrics']['f1_weighted']:.4f}")
    print(f"  Macro F1:    {test_results['metrics']['f1_macro']:.4f}")
    print(f"  95% CI:      [{test_results['accuracy_ci']['lower']*100:.2f}%, "
          f"{test_results['accuracy_ci']['upper']*100:.2f}%]")

    # Per-class metrics
    print(f"\nPer-class metrics (test set):")
    for class_name in CLASS_NAMES:
        metrics = test_results['metrics']['per_class'][class_name]
        print(f"  {class_name:8s}: P={metrics['precision']:.3f}, "
              f"R={metrics['recall']:.3f}, F1={metrics['f1']:.3f}, "
              f"N={metrics['support']}")

    # =========================================================================
    # 4. Save model and results
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 4: Save model and results")
    print("=" * 60)

    # Save model
    model_path = save_model(best_model, output_dir, 'svm')
    print(f"Model saved to: {model_path}")

    # Save results (legacy format)
    save_results(test_results, output_dir, 'svm_test')
    save_results(train_results, output_dir, 'svm_train')

    # Save test_metrics.json (RGB-aligned format)
    with open(output_dir / 'test_metrics.json', 'w') as f:
        json.dump(test_results['metrics'], f, indent=2)
    print(f"Metrics saved to: {output_dir / 'test_metrics.json'}")

    # Save predictions in RGB format (with probabilities)
    save_predictions(
        y_true=y_test,
        y_pred=test_results['predictions'],
        unique_ids=ids_test,
        output_path=output_dir / 'test_predictions.csv',
        probabilities=test_results['probabilities'],
        class_names=CLASS_NAMES
    )
    print(f"Predictions saved to: {output_dir / 'test_predictions.csv'}")

    # Save classification report (NEW - RGB aligned)
    save_classification_report(
        y_true=y_test,
        y_pred=test_results['predictions'],
        output_path=output_dir / 'classification_report.txt',
        class_names=CLASS_NAMES
    )
    print(f"Classification report saved to: {output_dir / 'classification_report.txt'}")

    # =========================================================================
    # 5. Generate visualizations
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 5: Generate visualizations")
    print("=" * 60)

    # Confusion matrix
    cm = compute_confusion_matrix(y_test, test_results['predictions'])

    # Save raw confusion matrix (NEW - RGB aligned)
    np.save(output_dir / 'confusion_matrix.npy', cm)
    print(f"Confusion matrix (npy) saved to: {output_dir / 'confusion_matrix.npy'}")

    # Save confusion matrix plot
    plot_confusion_matrix(
        cm,
        class_names=CLASS_NAMES,
        title='SVM - Test Set Confusion Matrix',
        save_path=output_dir / 'confusion_matrix.png',
        normalize=True,
        show_counts=True
    )
    print(f"Confusion matrix (png) saved to: {output_dir / 'confusion_matrix.png'}")

    # =========================================================================
    # 6. Generate LaTeX tables
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 6: Generate LaTeX tables")
    print("=" * 60)

    latex_table = per_class_to_latex(
        test_results['metrics'],
        'SVM',
        caption='Per-Class Classification Performance (SVM with 5 Statistical Features)',
        label='tab:svm_per_class'
    )

    with open(output_dir / 'per_class_table.tex', 'w') as f:
        f.write(latex_table)
    print(f"LaTeX table saved to: {output_dir / 'per_class_table.tex'}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nSVM Results Summary:")
    print(f"  Features: {SVM_FEATURES}")
    print(f"  Best params: {grid_results['best_params']}")
    print(f"  Test Accuracy: {test_results['metrics']['accuracy']*100:.2f}%")
    print(f"  Test Weighted F1: {test_results['metrics']['f1_weighted']:.4f}")
    print(f"\nAll outputs saved to: {output_dir}")

    return test_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train SVM baseline classifier',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with full grid search (default)
  python train.py

  # Skip full tuning, use reduced grid search (fast mode)
  python train.py --skip-tuning
        """
    )
    # Hyperparameter tuning options (aligned with other thermal classifiers)
    parser.add_argument('--skip-tuning', '--fast', action='store_true',
                        dest='skip_tuning',
                        help='Skip full grid search, use reduced parameter grid')
    parser.add_argument('--n-trials', type=int, default=OPTUNA_N_TRIALS,
                        help='Number of trials (ignored - SVM uses grid search)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for results')
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None
    main(
        skip_tuning=args.skip_tuning,
        n_trials=args.n_trials,
        output_dir=output_dir
    )
