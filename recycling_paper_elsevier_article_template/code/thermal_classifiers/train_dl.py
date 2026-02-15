#!/usr/bin/env python3
"""
Unified training script for deep learning thermal classifiers.

Handles: BiLSTM, BiGRU, Transformer, TCN, 1D CNN, InceptionTime.

Pipeline:
1. Load thermal data (train/val/test)
2. Optuna hyperparameter tuning (optional)
3. Train final model with best parameters
4. Evaluate on test set
5. Save results (metrics, predictions, confusion matrix)

Usage:
    python train_dl.py --method lstm
    python train_dl.py --method tcn --skip-tuning
    python train_dl.py --method transformer --n-trials 30
"""

import argparse
import json
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_DIR))

import numpy as np
import torch

from thermal_classifiers.shared.config import (
    RANDOM_SEED, NUM_CLASSES, INPUT_SIZE, MAX_SEQ_LEN,
    MAX_EPOCHS, EARLY_STOPPING_PATIENCE, BATCH_SIZE, WEIGHT_DECAY,
    OPTUNA_N_TRIALS, RESULTS_DIR,
    set_seed, get_device, print_config, validate_paths
)
from thermal_classifiers.shared.dataset import create_data_loaders, collate_fn
from thermal_classifiers.shared.training import train_model, evaluate
from thermal_classifiers.shared.evaluation import (
    compute_metrics, compute_confusion_matrix, plot_confusion_matrix,
    save_predictions, save_classification_report, save_metrics_json
)
from thermal_classifiers.shared.optimization import (
    run_optimization, load_best_params
)


# Model factory functions
MODEL_FACTORIES = {
    'lstm': lambda: __import__('thermal_classifiers.lstm.model', fromlist=['create_lstm_model']).create_lstm_model,
    'gru': lambda: __import__('thermal_classifiers.gru.model', fromlist=['create_gru_model']).create_gru_model,
    'transformer': lambda: __import__('thermal_classifiers.transformer.model', fromlist=['create_transformer_model']).create_transformer_model,
    'tcn': lambda: __import__('thermal_classifiers.tcn.model', fromlist=['create_tcn_model']).create_tcn_model,
    'cnn_1d': lambda: __import__('thermal_classifiers.cnn_1d.model', fromlist=['create_cnn_1d_model']).create_cnn_1d_model,
    'inceptiontime': lambda: __import__('thermal_classifiers.inceptiontime.model', fromlist=['create_inceptiontime_model']).create_inceptiontime_model,
}

# Default hyperparameters per method (used when --skip-tuning and no saved params)
DEFAULT_PARAMS = {
    'lstm': {'hidden_dim': 128, 'num_layers': 1, 'dropout': 0.3, 'learning_rate': 1e-3, 'pooling': 'mean'},
    'gru': {'hidden_dim': 128, 'num_layers': 2, 'dropout': 0.3, 'learning_rate': 1e-3, 'pooling': 'mean'},
    'transformer': {'d_model': 64, 'nhead': 4, 'num_layers': 4, 'dim_feedforward': 128, 'dropout': 0.3, 'learning_rate': 1e-4, 'pooling': 'cls'},
    'tcn': {'num_channels': 64, 'num_layers': 6, 'kernel_size': 3, 'dropout': 0.3, 'learning_rate': 1e-3, 'pooling': 'avg'},
    'cnn_1d': {'num_filters': 64, 'num_layers': 3, 'kernel_sizes_preset': 'decreasing', 'dropout': 0.3, 'learning_rate': 1e-3, 'pooling': 'last'},
    'inceptiontime': {'nb_filters': 32, 'depth': 6, 'kernel_sizes_preset': 'default', 'dropout': 0.2, 'learning_rate': 1e-3, 'use_residual': True},
}

# Kernel size presets for InceptionTime
INCEPTIONTIME_KERNEL_SIZES = {
    'small': [5, 10, 20],
    'default': [10, 20, 40],
    'large': [20, 40, 80],
    'uniform': [15, 15, 15],
}

# Kernel size presets for 1D CNN
CNN1D_KERNEL_SIZES = {
    'decreasing': [7, 5, 3],
    'uniform_small': [3, 3, 3],
    'uniform_medium': [5, 5, 5],
    'uniform_large': [7, 7, 7],
}


def create_model(method: str, params: dict, device: torch.device):
    """Create model from method name and hyperparameters."""
    create_fn = MODEL_FACTORIES[method]()

    if method == 'lstm':
        model = create_fn(
            input_size=INPUT_SIZE,
            hidden_dim=params.get('hidden_dim', 128),
            num_layers=params.get('num_layers', 1),
            num_classes=NUM_CLASSES,
            dropout=params.get('dropout', 0.3),
            bidirectional=True,
            pooling=params.get('pooling', 'mean')
        )
    elif method == 'gru':
        model = create_fn(
            input_size=INPUT_SIZE,
            hidden_dim=params.get('hidden_dim', 128),
            num_layers=params.get('num_layers', 2),
            num_classes=NUM_CLASSES,
            dropout=params.get('dropout', 0.3),
            bidirectional=True,
            pooling=params.get('pooling', 'mean')
        )
    elif method == 'transformer':
        model = create_fn(
            input_size=INPUT_SIZE,
            d_model=params.get('d_model', 64),
            nhead=params.get('nhead', 4),
            num_layers=params.get('num_layers', 4),
            dim_feedforward=params.get('dim_feedforward', 128),
            num_classes=NUM_CLASSES,
            dropout=params.get('dropout', 0.3),
            max_len=MAX_SEQ_LEN + 1,
            pooling=params.get('pooling', 'cls')
        )
    elif method == 'tcn':
        base = params.get('num_channels', 64)
        n_layers = params.get('num_layers', 6)
        model = create_fn(
            input_size=INPUT_SIZE,
            num_channels=[base] * n_layers,
            kernel_size=params.get('kernel_size', 3),
            num_classes=NUM_CLASSES,
            dropout=params.get('dropout', 0.3),
            pooling=params.get('pooling', 'avg')
        )
    elif method == 'cnn_1d':
        base = params.get('num_filters', 64)
        n_layers = params.get('num_layers', 3)
        num_filters = [base] + [base * 2] * (n_layers - 1)
        preset = params.get('kernel_sizes_preset', 'decreasing')
        kernel_sizes = CNN1D_KERNEL_SIZES.get(preset, [7, 5, 3])
        if n_layers <= len(kernel_sizes):
            kernel_sizes = kernel_sizes[:n_layers]
        else:
            kernel_sizes = kernel_sizes + [kernel_sizes[-1]] * (n_layers - len(kernel_sizes))
        model = create_fn(
            input_size=INPUT_SIZE,
            num_filters=num_filters,
            kernel_sizes=kernel_sizes,
            num_classes=NUM_CLASSES,
            dropout=params.get('dropout', 0.3),
            pooling=params.get('pooling', 'last'),
            use_batch_norm=True
        )
    elif method == 'inceptiontime':
        preset = params.get('kernel_sizes_preset', 'default')
        kernel_sizes = INCEPTIONTIME_KERNEL_SIZES.get(preset, [10, 20, 40])
        model = create_fn(
            input_size=INPUT_SIZE,
            num_classes=NUM_CLASSES,
            nb_filters=params.get('nb_filters', 32),
            depth=params.get('depth', 6),
            kernel_sizes=kernel_sizes,
            use_residual=params.get('use_residual', True),
            dropout=params.get('dropout', 0.2),
            use_ensemble=False
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    return model.to(device)


def train_and_evaluate(method: str, params: dict, device: torch.device, output_dir: Path):
    """Train model and evaluate on test set."""
    from thermal_classifiers.shared.config import CLASS_NAMES, IDX_TO_CLASS

    output_dir.mkdir(parents=True, exist_ok=True)

    # Create data loaders
    train_loader, val_loader, test_loader, class_weights = create_data_loaders(
        batch_size=BATCH_SIZE, verbose=True
    )

    # Create model
    model = create_model(method, params, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {method}")
    print(f"Parameters: {n_params:,}")

    # Train
    lr = params.get('learning_rate', 1e-3)
    results = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
        device=device,
        learning_rate=lr,
        max_epochs=MAX_EPOCHS,
        patience=EARLY_STOPPING_PATIENCE,
        weight_decay=WEIGHT_DECAY,
        checkpoint_dir=output_dir / 'checkpoints',
        model_name=method,
        verbose=True
    )

    print(f"\nBest val F1: {results['best_val_f1']:.4f} at epoch {results['best_epoch']}")

    # Save model
    torch.save(model.state_dict(), output_dir / 'model.pt')

    # Evaluate on test set
    import torch.nn as nn
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    eval_result = evaluate(model, test_loader, criterion, device,
                           return_predictions=True, compute_f1=True)
    y_true = eval_result['labels']
    y_pred = eval_result['predictions']
    unique_ids = eval_result.get('unique_ids', [])
    probabilities = eval_result.get('probabilities', None)

    # Compute detailed metrics
    metrics = compute_metrics(y_true, y_pred)

    print(f"\nTest Results:")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  F1 Weighted: {metrics['f1_weighted']:.4f}")
    print(f"  F1 Macro:    {metrics['f1_macro']:.4f}")
    print(f"\nPer-class F1:")
    for cls in CLASS_NAMES:
        pc = metrics['per_class'][cls]
        print(f"  {cls:8s}: P={pc['precision']:.3f} R={pc['recall']:.3f} F1={pc['f1']:.3f} (n={pc['support']})")

    # Save results
    save_metrics_json(metrics, output_dir / 'test_metrics.json')
    save_predictions(y_true, y_pred, unique_ids, output_dir / 'test_predictions.csv',
                     probabilities=probabilities)
    save_classification_report(y_true, y_pred, output_dir / 'classification_report.txt')

    cm = compute_confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, save_path=output_dir / 'confusion_matrix.png',
                          title=f'{method.upper()} - Confusion Matrix')
    np.save(output_dir / 'confusion_matrix.npy', cm)

    # Save hyperparameters used
    with open(output_dir / 'hyperparameters.json', 'w') as f:
        json.dump({'method': method, 'params': params, 'n_parameters': n_params}, f, indent=2)

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train Deep Learning Thermal Classifier')
    parser.add_argument('--method', type=str, required=True,
                        choices=list(MODEL_FACTORIES.keys()),
                        help='Model architecture to train')
    parser.add_argument('--skip-tuning', action='store_true',
                        help='Skip hyperparameter tuning, use saved/default params')
    parser.add_argument('--n-trials', type=int, default=OPTUNA_N_TRIALS,
                        help='Number of Optuna trials')
    args = parser.parse_args()

    set_seed(RANDOM_SEED)
    validate_paths()
    print_config()
    device = get_device()

    method = args.method
    output_dir = RESULTS_DIR / method

    print(f"\n{'='*70}")
    print(f"THERMAL DL TRAINING - {method.upper()}")
    print(f"{'='*70}")

    # Hyperparameter tuning or load existing
    if not args.skip_tuning:
        print(f"\nRunning Optuna tuning ({args.n_trials} trials)...")
        opt_results = run_optimization(method, n_trials=args.n_trials)
        params = opt_results.get('best_params', DEFAULT_PARAMS[method])
    else:
        # Try to load saved params
        params = load_best_params(method)
        if not params:
            params = DEFAULT_PARAMS[method]
            print(f"Using default parameters for {method}")
        else:
            print(f"Loaded tuned parameters for {method}")

    print(f"Parameters: {params}")

    # Train and evaluate
    metrics = train_and_evaluate(method, params, device, output_dir)

    print(f"\n{'='*70}")
    print(f"DONE: {method.upper()}")
    print(f"  F1 Macro:    {metrics['f1_macro']:.4f}")
    print(f"  F1 Weighted: {metrics['f1_weighted']:.4f}")
    print(f"  Results:     {output_dir}")
    print(f"{'='*70}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
