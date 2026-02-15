#!/usr/bin/env python3
"""
Ensemble evaluation script for thermal classification.

Loads pretrained TCN, BiLSTM, and 1D CNN models and evaluates ensemble performance.
No training required - this script only performs inference and evaluation.

Usage:
    python -m classifiers.thermal_classifiers.ensemble.train
    python -m classifiers.thermal_classifiers.ensemble.train --voting soft
    python -m classifiers.thermal_classifiers.ensemble.train --voting weighted
    python -m classifiers.thermal_classifiers.ensemble.train --optimize-weights
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np

CODE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(CODE_DIR))

import torch
import torch.nn.functional as F
from scipy.optimize import minimize

from thermal_classifiers.shared.config import (
    RANDOM_SEED, NUM_CLASSES, BATCH_SIZE, RESULTS_DIR, CLASS_NAMES,
    set_seed, get_device
)
from thermal_classifiers.shared.dataset import create_data_loaders
from thermal_classifiers.shared.evaluation import (
    compute_metrics, compute_confusion_matrix, plot_confusion_matrix,
    save_predictions, save_classification_report
)
from thermal_classifiers.ensemble.soft_voting import (
    ThermalEnsemble, load_default_ensemble, get_default_model_paths,
    soft_vote, weighted_vote, hard_vote
)


def evaluate_ensemble(
    ensemble: ThermalEnsemble,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device
) -> Dict:
    """
    Evaluate ensemble on a dataset.

    Args:
        ensemble: ThermalEnsemble instance
        data_loader: DataLoader to evaluate on
        device: Device for computation

    Returns:
        Dictionary with predictions, probabilities, labels, and unique_ids
    """
    ensemble.eval()

    all_preds = []
    all_probs = []
    all_labels = []
    all_ids = []
    individual_probs = {name: [] for name in ensemble.model_names}

    with torch.no_grad():
        for batch in data_loader:
            sequences, labels, lengths, masks, unique_ids = batch

            sequences = sequences.to(device)
            lengths = lengths.to(device)
            masks = masks.to(device)

            # Get ensemble and individual predictions
            ensemble_probs, ind_probs = ensemble.predict_proba(sequences, lengths, masks)

            preds = ensemble_probs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(ensemble_probs.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_ids.extend(unique_ids)

            for name in ensemble.model_names:
                individual_probs[name].extend(ind_probs[name].cpu().numpy())

    return {
        'predictions': np.array(all_preds),
        'probabilities': np.array(all_probs),
        'labels': np.array(all_labels),
        'unique_ids': all_ids,
        'individual_probs': {k: np.array(v) for k, v in individual_probs.items()}
    }


def optimize_ensemble_weights(
    individual_probs: Dict[str, np.ndarray],
    labels: np.ndarray,
    model_names: List[str]
) -> Tuple[List[float], float]:
    """
    Optimize ensemble weights on validation set to maximize F1 macro.

    Args:
        individual_probs: Dict of model_name -> [N, num_classes] probabilities
        labels: Ground truth labels [N]
        model_names: List of model names in order

    Returns:
        Tuple of (optimal_weights, best_f1)
    """
    from sklearn.metrics import f1_score

    def objective(weights):
        # Normalize weights
        weights = np.array(weights) / np.sum(weights)

        # Compute weighted ensemble predictions
        ensemble_probs = np.zeros_like(list(individual_probs.values())[0])
        for name, w in zip(model_names, weights):
            ensemble_probs += individual_probs[name] * w

        preds = ensemble_probs.argmax(axis=1)
        f1 = f1_score(labels, preds, average='macro')
        return -f1  # Minimize negative F1

    # Initial weights (equal)
    n_models = len(model_names)
    x0 = [1.0 / n_models] * n_models

    # Bounds: each weight between 0.1 and 0.6
    bounds = [(0.1, 0.6)] * n_models

    # Constraints: weights sum to 1
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}

    result = minimize(
        objective, x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 100}
    )

    optimal_weights = list(result.x / np.sum(result.x))
    best_f1 = -result.fun

    return optimal_weights, best_f1


def main(
    voting: str = 'weighted',
    optimize_weights: bool = False,
    output_dir: Path = None
):
    """Main ensemble evaluation pipeline."""
    print("=" * 60)
    print("THERMAL ENSEMBLE EVALUATION")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")

    set_seed(RANDOM_SEED)
    device = get_device()
    print(f"Device: {device}")

    if output_dir is None:
        output_dir = RESULTS_DIR / 'ensemble'
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Check model paths
    print("\n" + "=" * 60)
    print("Loading pretrained models...")
    paths = get_default_model_paths()
    for name, path in paths.items():
        exists = "✓" if path.exists() else "✗"
        print(f"  {name}: {exists} {path.name}")

    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        print(f"\nERROR: Missing model files: {missing}")
        print("Run ablation studies first to generate model files.")
        return None, None

    # Load data
    print("\n" + "=" * 60)
    print("Loading data...")
    _, val_loader, test_loader, _ = create_data_loaders(
        batch_size=BATCH_SIZE,
        augment_train=False,
        verbose=True
    )

    # Default weights (F1-proportional)
    default_weights = None  # Will use F1-proportional in load_default_ensemble

    # Optimize weights on validation set if requested
    if optimize_weights:
        print("\n" + "=" * 60)
        print("Optimizing ensemble weights on validation set...")

        # Load ensemble with equal weights for optimization
        ensemble = load_default_ensemble(device=device, weights=[1/3, 1/3, 1/3])

        # Evaluate on validation set
        val_results = evaluate_ensemble(ensemble, val_loader, device)

        # Optimize weights
        optimal_weights, best_val_f1 = optimize_ensemble_weights(
            val_results['individual_probs'],
            val_results['labels'],
            ensemble.model_names
        )

        print(f"Optimal weights found:")
        for name, weight in zip(ensemble.model_names, optimal_weights):
            print(f"  {name}: {weight:.4f}")
        print(f"Validation F1 with optimal weights: {best_val_f1:.4f}")

        # Reload with optimal weights
        ensemble = load_default_ensemble(device=device, weights=optimal_weights)
    else:
        # Load with default F1-proportional weights
        ensemble = load_default_ensemble(device=device)

    print(f"\nEnsemble configuration:")
    print(f"  Models: {ensemble.model_names}")
    print(f"  Weights: {[f'{w:.4f}' for w in ensemble.weights]}")
    print(f"  Total parameters: {ensemble.get_num_parameters():,}")
    print(f"  Voting method: {voting}")

    # Evaluate on test set
    print("\n" + "=" * 60)
    print("Evaluating on test set...")
    test_results = evaluate_ensemble(ensemble, test_loader, device)

    # Compute metrics
    test_metrics = compute_metrics(test_results['labels'], test_results['predictions'])

    print(f"\nTest Results:")
    print(f"  Accuracy:    {test_metrics['accuracy']*100:.2f}%")
    print(f"  Weighted F1: {test_metrics['f1_weighted']:.4f}")
    print(f"  Macro F1:    {test_metrics['f1_macro']:.4f}")

    # Compare with individual models
    print("\n" + "=" * 60)
    print("Comparison with individual models:")
    print(f"{'Model':<12} {'F1 Macro':<12} {'Δ vs Ensemble':<15}")
    print("-" * 40)

    individual_metrics = {}
    for name in ensemble.model_names:
        probs = test_results['individual_probs'][name]
        preds = probs.argmax(axis=1)
        metrics = compute_metrics(test_results['labels'], preds)
        individual_metrics[name] = metrics

        delta = test_metrics['f1_macro'] - metrics['f1_macro']
        sign = '+' if delta >= 0 else ''
        print(f"{name.upper():<12} {metrics['f1_macro']:.4f}       {sign}{delta*100:.2f} pp")

    print("-" * 40)
    print(f"{'ENSEMBLE':<12} {test_metrics['f1_macro']:.4f}")

    # Save results
    print("\n" + "=" * 60)
    print("Saving results...")

    # Test metrics
    metrics_to_save = {
        'model': 'ensemble',
        'voting_method': voting,
        'weights': dict(zip(ensemble.model_names, ensemble.weights)),
        'accuracy': float(test_metrics['accuracy']),
        'f1_weighted': float(test_metrics['f1_weighted']),
        'f1_macro': float(test_metrics['f1_macro']),
        'precision_weighted': float(test_metrics['precision_weighted']),
        'recall_weighted': float(test_metrics['recall_weighted']),
        'per_class': {k: {kk: float(vv) if isinstance(vv, (float, int)) else vv
                          for kk, vv in v.items()}
                      for k, v in test_metrics['per_class'].items()},
        'support': test_metrics.get('support', {}),
        'num_parameters': ensemble.get_num_parameters(),
        'individual_f1_macro': {
            name: float(individual_metrics[name]['f1_macro'])
            for name in ensemble.model_names
        }
    }
    with open(output_dir / 'test_metrics.json', 'w') as f:
        json.dump(metrics_to_save, f, indent=2)
    print(f"  Test metrics: {output_dir / 'test_metrics.json'}")

    # Predictions
    save_predictions(
        y_true=test_results['labels'],
        y_pred=test_results['predictions'],
        unique_ids=test_results['unique_ids'],
        output_path=output_dir / 'test_predictions.csv',
        probabilities=test_results['probabilities'],
        class_names=CLASS_NAMES
    )
    print(f"  Predictions: {output_dir / 'test_predictions.csv'}")

    # Classification report
    save_classification_report(
        y_true=test_results['labels'],
        y_pred=test_results['predictions'],
        output_path=output_dir / 'classification_report.txt',
        class_names=CLASS_NAMES
    )
    print(f"  Report: {output_dir / 'classification_report.txt'}")

    # Confusion matrix
    cm = compute_confusion_matrix(test_results['labels'], test_results['predictions'])
    np.save(output_dir / 'confusion_matrix.npy', cm)
    plot_confusion_matrix(
        cm, class_names=CLASS_NAMES,
        title='Ensemble - Test Set Confusion Matrix',
        save_path=output_dir / 'confusion_matrix.png',
        normalize=True, show_counts=True
    )
    print(f"  Confusion matrix: {output_dir / 'confusion_matrix.png'}")

    # Hyperparameters / config
    config_to_save = {
        'ensemble_type': 'soft_voting',
        'voting_method': voting,
        'models': ensemble.model_names,
        'weights': dict(zip(ensemble.model_names, ensemble.weights)),
        'weights_optimized': optimize_weights,
        'model_paths': {name: str(paths[name]) for name in ensemble.model_names},
        'individual_f1_macro': {
            name: float(individual_metrics[name]['f1_macro'])
            for name in ensemble.model_names
        }
    }
    with open(output_dir / 'hyperparameters.json', 'w') as f:
        json.dump(config_to_save, f, indent=2)

    # Also save as best_params.json for compatibility
    with open(output_dir / 'best_params.json', 'w') as f:
        json.dump(config_to_save, f, indent=2)

    print("\n" + "=" * 60)
    print("ENSEMBLE EVALUATION COMPLETE")
    print(f"All outputs saved to: {output_dir}")

    return test_metrics, test_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate thermal ensemble')
    parser.add_argument('--voting', type=str, default='weighted',
                        choices=['soft', 'weighted', 'hard'],
                        help='Voting method (soft=equal, weighted=F1-proportional)')
    parser.add_argument('--optimize-weights', action='store_true',
                        help='Optimize weights on validation set')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory')
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None
    main(voting=args.voting, optimize_weights=args.optimize_weights, output_dir=output_dir)
