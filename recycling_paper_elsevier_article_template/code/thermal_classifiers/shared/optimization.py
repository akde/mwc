#!/usr/bin/env python3
"""
Optuna-based hyperparameter tuning for thermal classifiers.

Supports: SVM, BiLSTM, BiGRU, Transformer, TCN

Usage:
    python hyperparameter_tuning.py --model lstm --n-trials 50
    python hyperparameter_tuning.py --model all --n-trials 30
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import json
import warnings
import sqlite3

CODE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(CODE_DIR))

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: Optuna not installed. Run: pip install optuna")

from thermal_classifiers.shared.config import (
    RANDOM_SEED, NUM_CLASSES, INPUT_SIZE, BATCH_SIZE,
    OPTUNA_N_TRIALS, OPTUNA_CV_FOLDS, OPTUNA_TUNING_EPOCHS,
    BEST_PARAMS_DIR, set_seed, get_device
)
from thermal_classifiers.shared.dataset import (
    create_data_loaders, load_data, compute_class_weights_tensor
)
from thermal_classifiers.shared.training import (
    EarlyStopping, train_epoch, evaluate
)

# Suppress warnings during optimization
warnings.filterwarnings('ignore')


def create_svm_objective(X_train, y_train, cv_folds=OPTUNA_CV_FOLDS):
    """Create Optuna objective for SVM hyperparameter tuning."""

    def objective(trial):
        # Suggest hyperparameters
        C = trial.suggest_float('C', 0.1, 100.0, log=True)
        gamma = trial.suggest_categorical('gamma', ['scale', 'auto', 0.001, 0.01, 0.1, 1.0])
        kernel = trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid'])

        # Create pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(C=C, gamma=gamma, kernel=kernel, class_weight='balanced',
                       random_state=RANDOM_SEED))
        ])

        # Cross-validation
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv_folds,
                                  scoring='f1_weighted', n_jobs=-1)

        return scores.mean()

    return objective


def create_lstm_objective(train_loader, val_loader, class_weights, device):
    """Create Optuna objective for BiLSTM hyperparameter tuning."""
    from thermal_classifiers.lstm.model import create_lstm_model

    def objective(trial):
        # Suggest hyperparameters
        hidden_dim = trial.suggest_categorical('hidden_dim', [32, 64, 128, 256])
        num_layers = trial.suggest_int('num_layers', 1, 3)
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        pooling = trial.suggest_categorical('pooling', ['last', 'mean'])

        # Create model
        model = create_lstm_model(
            input_size=INPUT_SIZE,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=NUM_CLASSES,
            dropout=dropout,
            bidirectional=True,
            pooling=pooling
        ).to(device)

        # Training setup
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

        # Train for limited epochs
        best_val_f1 = 0.0
        for epoch in range(OPTUNA_TUNING_EPOCHS):
            train_epoch(model, train_loader, optimizer, criterion, device, log_interval=0)
            val_results = evaluate(model, val_loader, criterion, device, return_predictions=True)

            # Compute F1
            from sklearn.metrics import f1_score
            val_f1 = f1_score(val_results['labels'], val_results['predictions'],
                             average='weighted', zero_division=0)
            best_val_f1 = max(best_val_f1, val_f1)

            # Pruning
            trial.report(val_f1, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return best_val_f1

    return objective


def create_gru_objective(train_loader, val_loader, class_weights, device):
    """Create Optuna objective for BiGRU hyperparameter tuning."""
    from thermal_classifiers.gru.model import create_gru_model

    def objective(trial):
        hidden_dim = trial.suggest_categorical('hidden_dim', [32, 64, 128, 256])
        num_layers = trial.suggest_int('num_layers', 1, 3)
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        pooling = trial.suggest_categorical('pooling', ['last', 'mean'])

        model = create_gru_model(
            input_size=INPUT_SIZE,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=NUM_CLASSES,
            dropout=dropout,
            bidirectional=True,
            pooling=pooling
        ).to(device)

        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

        best_val_f1 = 0.0
        for epoch in range(OPTUNA_TUNING_EPOCHS):
            train_epoch(model, train_loader, optimizer, criterion, device, log_interval=0)
            val_results = evaluate(model, val_loader, criterion, device, return_predictions=True)

            from sklearn.metrics import f1_score
            val_f1 = f1_score(val_results['labels'], val_results['predictions'],
                             average='weighted', zero_division=0)
            best_val_f1 = max(best_val_f1, val_f1)

            trial.report(val_f1, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return best_val_f1

    return objective


def create_transformer_objective(train_loader, val_loader, class_weights, device):
    """Create Optuna objective for Transformer hyperparameter tuning."""
    from thermal_classifiers.transformer.model import create_transformer_model
    from thermal_classifiers.shared.config import MAX_SEQ_LEN

    def objective(trial):
        d_model = trial.suggest_categorical('d_model', [32, 64, 128])
        # nhead must divide d_model
        valid_nheads = [h for h in [2, 4, 8] if d_model % h == 0]
        nhead = trial.suggest_categorical('nhead', valid_nheads)
        num_layers = trial.suggest_int('num_layers', 2, 6)
        dim_feedforward = trial.suggest_categorical('dim_feedforward', [64, 128, 256])
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
        pooling = trial.suggest_categorical('pooling', ['cls', 'mean'])

        model = create_transformer_model(
            input_size=INPUT_SIZE,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            num_classes=NUM_CLASSES,
            dropout=dropout,
            max_len=MAX_SEQ_LEN + 1,
            pooling=pooling
        ).to(device)

        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

        best_val_f1 = 0.0
        for epoch in range(OPTUNA_TUNING_EPOCHS):
            train_epoch(model, train_loader, optimizer, criterion, device, log_interval=0)
            val_results = evaluate(model, val_loader, criterion, device, return_predictions=True)

            from sklearn.metrics import f1_score
            val_f1 = f1_score(val_results['labels'], val_results['predictions'],
                             average='weighted', zero_division=0)
            best_val_f1 = max(best_val_f1, val_f1)

            trial.report(val_f1, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return best_val_f1

    return objective


def create_tcn_objective(train_loader, val_loader, class_weights, device):
    """Create Optuna objective for TCN hyperparameter tuning."""
    from thermal_classifiers.tcn.model import create_tcn_model

    def objective(trial):
        num_channels_base = trial.suggest_categorical('num_channels', [32, 64, 128])
        num_layers = trial.suggest_int('num_layers', 4, 8)
        num_channels = [num_channels_base] * num_layers
        kernel_size = trial.suggest_categorical('kernel_size', [3, 5, 7])
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        pooling = trial.suggest_categorical('pooling', ['avg', 'max', 'last'])

        model = create_tcn_model(
            input_size=INPUT_SIZE,
            num_channels=num_channels,
            kernel_size=kernel_size,
            num_classes=NUM_CLASSES,
            dropout=dropout,
            pooling=pooling
        ).to(device)

        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

        best_val_f1 = 0.0
        for epoch in range(OPTUNA_TUNING_EPOCHS):
            train_epoch(model, train_loader, optimizer, criterion, device, log_interval=0)
            val_results = evaluate(model, val_loader, criterion, device, return_predictions=True)

            from sklearn.metrics import f1_score
            val_f1 = f1_score(val_results['labels'], val_results['predictions'],
                             average='weighted', zero_division=0)
            best_val_f1 = max(best_val_f1, val_f1)

            trial.report(val_f1, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return best_val_f1

    return objective


def create_cnn_1d_objective(train_loader, val_loader, class_weights, device):
    """
    Create Optuna objective for 1D CNN hyperparameter tuning.

    Tunes: num_filters (base), num_layers, kernel_sizes, dropout, pooling, learning_rate
    """
    from thermal_classifiers.cnn_1d.model import create_cnn_1d_model

    # Pre-defined kernel size patterns
    KERNEL_SIZES_OPTIONS = {
        'decreasing': [7, 5, 3],
        'uniform_small': [3, 3, 3],
        'uniform_medium': [5, 5, 5],
        'uniform_large': [7, 7, 7],
    }

    def objective(trial):
        # Architecture hyperparameters
        num_filters_base = trial.suggest_categorical('num_filters', [32, 64, 128])
        num_layers = trial.suggest_int('num_layers', 2, 4)
        kernel_sizes_preset = trial.suggest_categorical('kernel_sizes_preset',
                                                         list(KERNEL_SIZES_OPTIONS.keys()))

        # Build filter and kernel lists for num_layers
        # Filters: [base, base*2, base*2, ...] (double after first layer)
        num_filters = [num_filters_base] + [num_filters_base * 2] * (num_layers - 1)

        # Kernel sizes: use preset pattern, extend/truncate to num_layers
        base_kernels = KERNEL_SIZES_OPTIONS[kernel_sizes_preset]
        if num_layers <= len(base_kernels):
            kernel_sizes = base_kernels[:num_layers]
        else:
            kernel_sizes = base_kernels + [base_kernels[-1]] * (num_layers - len(base_kernels))

        # Regularization
        dropout = trial.suggest_float('dropout', 0.1, 0.5)

        # Pooling strategy
        pooling = trial.suggest_categorical('pooling', ['last', 'avg', 'max'])

        # Optimizer
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)

        # Create model
        model = create_cnn_1d_model(
            input_size=INPUT_SIZE,
            num_filters=num_filters,
            kernel_sizes=kernel_sizes,
            num_classes=NUM_CLASSES,
            dropout=dropout,
            pooling=pooling,
            use_batch_norm=True
        ).to(device)

        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

        best_val_f1 = 0.0
        for epoch in range(OPTUNA_TUNING_EPOCHS):
            train_epoch(model, train_loader, optimizer, criterion, device, log_interval=0)
            val_results = evaluate(model, val_loader, criterion, device, return_predictions=True)

            from sklearn.metrics import f1_score
            val_f1 = f1_score(val_results['labels'], val_results['predictions'],
                             average='weighted', zero_division=0)
            best_val_f1 = max(best_val_f1, val_f1)

            trial.report(val_f1, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return best_val_f1

    return objective


# =============================================================================
# NEW OBJECTIVES FOR: tsfresh_xgboost, lite, ts2vec, distillation
# =============================================================================

def create_tsfresh_xgboost_objective(X_train, y_train, cv_folds=OPTUNA_CV_FOLDS):
    """
    Create Optuna objective for TSFresh + XGBoost hyperparameter tuning.

    Args:
        X_train: Pre-extracted TSFresh features (expensive extraction done once)
        y_train: Labels
        cv_folds: Number of cross-validation folds

    Note: TSFresh feature extraction takes ~5 min, so features should be
    extracted ONCE before calling this function.
    """
    import xgboost as xgb

    def objective(trial):
        # Core XGBoost params
        n_estimators = trial.suggest_int('n_estimators', 50, 500)
        max_depth = trial.suggest_int('max_depth', 2, 8)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)

        # Regularization params (critical for 800+ TSFresh features)
        min_child_weight = trial.suggest_int('min_child_weight', 1, 10)
        subsample = trial.suggest_float('subsample', 0.6, 1.0)
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.6, 1.0)
        reg_alpha = trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True)
        reg_lambda = trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True)

        clf = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            objective='multi:softprob',
            random_state=RANDOM_SEED,
            n_jobs=-1,
            eval_metric='mlogloss',
            use_label_encoder=False
        )

        # Cross-validation
        scores = cross_val_score(clf, X_train, y_train, cv=cv_folds,
                                  scoring='f1_weighted', n_jobs=-1)

        return scores.mean()

    return objective


def create_lite_objective(train_loader, val_loader, class_weights, device):
    """Create Optuna objective for LITE classifier hyperparameter tuning."""
    from thermal_classifiers.lite.model import create_lite_model

    def objective(trial):
        # n_filters must be divisible by 6 (number of kernel groups in LITE)
        n_filters = trial.suggest_categorical('n_filters', [24, 30, 36, 48, 60])
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        pooling = trial.suggest_categorical('pooling', ['avg', 'max', 'last'])
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        use_custom_filters = trial.suggest_categorical('use_custom_filters', [True, False])

        model = create_lite_model(
            input_size=INPUT_SIZE,
            num_classes=NUM_CLASSES,
            n_filters=n_filters,
            use_custom_filters=use_custom_filters,
            dropout=dropout,
            pooling=pooling
        ).to(device)

        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

        best_val_f1 = 0.0
        for epoch in range(OPTUNA_TUNING_EPOCHS):
            train_epoch(model, train_loader, optimizer, criterion, device, log_interval=0)
            val_results = evaluate(model, val_loader, criterion, device, return_predictions=True)

            from sklearn.metrics import f1_score
            val_f1 = f1_score(val_results['labels'], val_results['predictions'],
                             average='weighted', zero_division=0)
            best_val_f1 = max(best_val_f1, val_f1)

            trial.report(val_f1, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return best_val_f1

    return objective


def create_ts2vec_objective(train_seqs, val_seqs, train_labels, val_labels, device, max_train_length=1000):
    """
    Create Optuna objective for TS2Vec + Linear classifier hyperparameter tuning.

    Args:
        train_seqs: Training sequences (padded numpy array [N, T, 1])
        val_seqs: Validation sequences (padded numpy array [N, T, 1])
        train_labels: Training labels
        val_labels: Validation labels
        device: torch device
        max_train_length: Maximum sequence length for TS2Vec

    Note: TS2Vec pretraining is expensive, so pretrain_epochs is reduced during tuning.
    """
    from ts2vec import TS2Vec
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score

    def objective(trial):
        # Encoder architecture
        output_dims = trial.suggest_categorical('output_dims', [64, 128, 256])
        hidden_dims = trial.suggest_categorical('hidden_dims', [32, 64, 128])
        depth = trial.suggest_int('depth', 4, 12)

        # Pretraining (reduced during tuning for speed)
        pretrain_epochs = trial.suggest_int('pretrain_epochs', 20, 60)

        # Classifier regularization
        classifier_C = trial.suggest_float('classifier_C', 0.01, 100.0, log=True)

        # Create and train TS2Vec encoder
        model = TS2Vec(
            input_dims=1,
            output_dims=output_dims,
            hidden_dims=hidden_dims,
            depth=depth,
            device=device,
            lr=0.001,
            batch_size=16,
            max_train_length=max_train_length
        )

        # Combine train + val for self-supervised pretraining
        pretrain_data = np.vstack([train_seqs, val_seqs])
        model.fit(pretrain_data, n_epochs=pretrain_epochs, verbose=False)

        # Extract representations
        train_repr = model.encode(train_seqs, encoding_window='full_series')
        val_repr = model.encode(val_seqs, encoding_window='full_series')

        # Normalize
        scaler = StandardScaler()
        train_repr_scaled = scaler.fit_transform(train_repr)
        val_repr_scaled = scaler.transform(val_repr)

        # Train linear classifier
        classifier = LogisticRegression(
            C=classifier_C,
            max_iter=1000,
            class_weight='balanced',
            random_state=RANDOM_SEED,
            solver='lbfgs',
            multi_class='multinomial'
        )
        classifier.fit(train_repr_scaled, train_labels)

        # Evaluate
        val_pred = classifier.predict(val_repr_scaled)
        val_f1 = f1_score(val_labels, val_pred, average='weighted', zero_division=0)

        return val_f1

    return objective


def create_distillation_objective(train_loader, val_loader, teacher, class_weights, device):
    """
    Create Optuna objective for knowledge distillation hyperparameter tuning.

    Args:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        teacher: Pre-loaded teacher model (frozen, in eval mode)
        class_weights: Class weights tensor
        device: torch device

    Note: Teacher model should be loaded ONCE before calling this function
    to avoid expensive re-loading per trial.
    """
    from thermal_classifiers.tcn.model import create_tcn_model
    from thermal_classifiers.distillation.train import (
        DistillationLoss, train_distillation_epoch, evaluate_student
    )

    def objective(trial):
        # Distillation params
        temperature = trial.suggest_float('temperature', 1.0, 10.0)
        alpha = trial.suggest_float('alpha', 0.1, 0.9)

        # Student architecture
        num_layers = trial.suggest_int('num_layers', 2, 5)
        base_channels = trial.suggest_categorical('base_channels', [16, 32, 48, 64])
        student_channels = [base_channels] * num_layers
        student_kernel_size = trial.suggest_categorical('student_kernel_size', [3, 5, 7])

        # Training
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)

        # Create student
        student = create_tcn_model(
            input_size=INPUT_SIZE,
            num_channels=student_channels,
            kernel_size=student_kernel_size,
            num_classes=NUM_CLASSES,
            dropout=0.2,
            pooling='avg'
        ).to(device)

        # Distillation loss
        criterion = DistillationLoss(
            temperature=temperature,
            alpha=alpha,
            class_weights=class_weights.to(device)
        )

        # Optimizer
        optimizer = torch.optim.AdamW(student.parameters(), lr=learning_rate, weight_decay=0.01)

        best_val_f1 = 0.0
        for epoch in range(OPTUNA_TUNING_EPOCHS):
            train_distillation_epoch(student, teacher, train_loader, optimizer, criterion, device)
            val_results = evaluate_student(student, val_loader, criterion, device, return_predictions=True)

            val_f1 = val_results.get('f1_weighted', 0.0)
            best_val_f1 = max(best_val_f1, val_f1)

            trial.report(val_f1, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return best_val_f1

    return objective


def create_inceptiontime_objective(train_loader, val_loader, class_weights, device):
    """
    Create Optuna objective for InceptionTime hyperparameter tuning.

    Args:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        class_weights: Class weights tensor for imbalanced classes
        device: torch device

    Returns:
        Callable objective function for Optuna study

    Note:
        - Ensemble mode is disabled during tuning (5x training cost)
        - depth must be multiple of 3 (enforced via categorical choices)
        - kernel_sizes uses pre-defined triplets for valid multi-scale patterns
    """
    from thermal_classifiers.inceptiontime.model import create_inceptiontime_model

    # Pre-defined kernel size triplets (ensures valid multi-scale patterns)
    KERNEL_SIZES_OPTIONS = {
        'small': [5, 10, 20],
        'default': [10, 20, 40],
        'large': [20, 40, 80],
        'uniform': [15, 15, 15],
    }

    def objective(trial):
        # Architecture hyperparameters
        nb_filters = trial.suggest_categorical('nb_filters', [16, 32, 48, 64])
        depth = trial.suggest_categorical('depth', [3, 6, 9, 12])  # Must be multiple of 3
        kernel_sizes_preset = trial.suggest_categorical('kernel_sizes_preset',
                                                         list(KERNEL_SIZES_OPTIONS.keys()))
        kernel_sizes = KERNEL_SIZES_OPTIONS[kernel_sizes_preset]
        use_residual = trial.suggest_categorical('use_residual', [True, False])

        # Regularization
        dropout = trial.suggest_float('dropout', 0.0, 0.5)
        label_smoothing = trial.suggest_float('label_smoothing', 0.0, 0.2)

        # Optimizer
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)

        # Create model (ensemble disabled during tuning)
        model = create_inceptiontime_model(
            input_size=INPUT_SIZE,
            num_classes=NUM_CLASSES,
            nb_filters=nb_filters,
            depth=depth,
            kernel_sizes=kernel_sizes,
            use_residual=use_residual,
            dropout=dropout,
            use_ensemble=False  # Always single network during tuning
        ).to(device)

        # Loss with label smoothing
        criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(device),
            label_smoothing=label_smoothing
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

        # Train for limited epochs with pruning
        best_val_f1 = 0.0
        for epoch in range(OPTUNA_TUNING_EPOCHS):
            train_epoch(model, train_loader, optimizer, criterion, device, log_interval=0)
            val_results = evaluate(model, val_loader, criterion, device, return_predictions=True)

            from sklearn.metrics import f1_score
            val_f1 = f1_score(val_results['labels'], val_results['predictions'],
                             average='weighted', zero_division=0)
            best_val_f1 = max(best_val_f1, val_f1)

            # Report to Optuna for pruning
            trial.report(val_f1, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return best_val_f1

    return objective


def run_optimization(
    model_type: str,
    n_trials: int = OPTUNA_N_TRIALS,
    output_dir: Path = None
) -> dict:
    """
    Run Optuna optimization for a specific model type.

    Args:
        model_type: One of 'svm', 'lstm', 'gru', 'transformer', 'tcn',
                    'tsfresh_xgboost', 'lite', 'ts2vec', 'distillation'
        n_trials: Number of optimization trials
        output_dir: Directory to save results

    Returns:
        Dict with best_params and best_score
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna not installed. Run: pip install optuna")

    set_seed(RANDOM_SEED)

    if output_dir is None:
        output_dir = BEST_PARAMS_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"\n{'='*60}")
    print(f"OPTUNA HYPERPARAMETER TUNING: {model_type.upper()}")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Trials: {n_trials}")
    print(f"Epochs per trial: {OPTUNA_TUNING_EPOCHS}")

    # Create study with SQLite storage for crash recovery
    sampler = TPESampler(seed=RANDOM_SEED)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=5)

    # SQLite storage for crash recovery and trial tracking
    db_path = output_dir / 'optuna_study.db'
    storage = f'sqlite:///{db_path}'

    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        pruner=pruner,
        study_name=f'thermal_{model_type}',
        storage=storage,
        load_if_exists=True
    )

    # Check for existing trials (resume capability)
    existing_trials = len(study.trials)
    if existing_trials > 0:
        print(f"Resuming from {existing_trials} existing trials")

    # Create objective based on model type
    if model_type == 'svm':
        from thermal_classifiers.shared.feature_extraction import load_and_extract_features
        from thermal_classifiers.shared.config import TRAIN_DATA_PATH, VAL_DATA_PATH

        X_train, y_train, _ = load_and_extract_features(str(TRAIN_DATA_PATH), verbose=False)
        X_val, y_val, _ = load_and_extract_features(str(VAL_DATA_PATH), verbose=False)
        X_combined = np.vstack([X_train, X_val])
        y_combined = np.concatenate([y_train, y_val])

        objective = create_svm_objective(X_combined, y_combined)

    elif model_type == 'tsfresh_xgboost':
        # TSFresh feature extraction is expensive (~5 min), extract ONCE
        from thermal_classifiers.tsfresh_xgboost.feature_extraction import (
            extract_tsfresh_features, impute_features
        )
        from thermal_classifiers.shared.config import TRAIN_DATA_PATH, VAL_DATA_PATH

        print("Extracting TSFresh features (this takes ~5 minutes)...")
        X_train, y_train = extract_tsfresh_features(str(TRAIN_DATA_PATH))
        X_val, y_val = extract_tsfresh_features(str(VAL_DATA_PATH))

        # Impute NaN values (TSFresh can produce NaNs for some features)
        X_train = impute_features(X_train)
        X_val = impute_features(X_val)

        X_combined = np.vstack([X_train, X_val])
        y_combined = np.concatenate([y_train, y_val])

        print(f"TSFresh features extracted: {X_combined.shape[1]} features")
        objective = create_tsfresh_xgboost_objective(X_combined, y_combined)

    elif model_type == 'ts2vec_ft':
        # TS2Vec uses raw sequences for self-supervised pretraining
        from thermal_classifiers.shared.config import TRAIN_DATA_PATH, VAL_DATA_PATH

        # Load raw sequences (not DataLoaders)
        # load_data returns tuple: (sequences, labels, unique_ids)
        train_seqs_raw, train_labels_raw, _ = load_data(str(TRAIN_DATA_PATH))
        val_seqs_raw, val_labels_raw, _ = load_data(str(VAL_DATA_PATH))

        # Convert to numpy arrays with shape [N, T, 1] for TS2Vec
        train_seqs = np.array([np.array(s).reshape(-1, 1) for s in train_seqs_raw], dtype=object)
        val_seqs = np.array([np.array(s).reshape(-1, 1) for s in val_seqs_raw], dtype=object)
        train_labels = np.array(train_labels_raw)
        val_labels = np.array(val_labels_raw)

        # Pad to uniform length for TS2Vec
        max_len = max(max(len(s) for s in train_seqs), max(len(s) for s in val_seqs))
        train_seqs_padded = np.zeros((len(train_seqs), max_len, 1))
        val_seqs_padded = np.zeros((len(val_seqs), max_len, 1))
        for i, s in enumerate(train_seqs):
            train_seqs_padded[i, :len(s), :] = s
        for i, s in enumerate(val_seqs):
            val_seqs_padded[i, :len(s), :] = s

        print(f"TS2Vec data: {len(train_seqs)} train, {len(val_seqs)} val, max_len={max_len}")
        objective = create_ts2vec_objective(
            train_seqs_padded, val_seqs_padded, train_labels, val_labels, device
        )

    elif model_type == 'distillation':
        # Load teacher model ONCE (expensive to re-load per trial)
        from thermal_classifiers.distillation.train import load_default_ensemble

        train_loader, val_loader, _, class_weights = create_data_loaders(
            batch_size=BATCH_SIZE, verbose=False
        )

        print("Loading teacher ensemble (this may take a moment)...")
        teacher = load_default_ensemble(device)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False
        print(f"Teacher loaded and frozen")

        objective = create_distillation_objective(
            train_loader, val_loader, teacher, class_weights, device
        )

    elif model_type == 'lite':
        train_loader, val_loader, _, class_weights = create_data_loaders(
            batch_size=BATCH_SIZE, verbose=False
        )
        objective = create_lite_objective(train_loader, val_loader, class_weights, device)

    elif model_type == 'inceptiontime':
        train_loader, val_loader, _, class_weights = create_data_loaders(
            batch_size=BATCH_SIZE, verbose=False
        )
        objective = create_inceptiontime_objective(train_loader, val_loader, class_weights, device)

    elif model_type in ('lstm', 'gru', 'transformer', 'tcn', 'cnn_1d'):
        train_loader, val_loader, _, class_weights = create_data_loaders(
            batch_size=BATCH_SIZE, verbose=False
        )

        if model_type == 'lstm':
            objective = create_lstm_objective(train_loader, val_loader, class_weights, device)
        elif model_type == 'gru':
            objective = create_gru_objective(train_loader, val_loader, class_weights, device)
        elif model_type == 'transformer':
            objective = create_transformer_objective(train_loader, val_loader, class_weights, device)
        elif model_type == 'tcn':
            objective = create_tcn_objective(train_loader, val_loader, class_weights, device)
        elif model_type == 'cnn_1d':
            objective = create_cnn_1d_objective(train_loader, val_loader, class_weights, device)

    else:
        raise ValueError(f"Unknown model type: {model_type}. "
                         f"Supported: svm, lstm, gru, transformer, tcn, cnn_1d, "
                         f"tsfresh_xgboost, lite, ts2vec_ft, distillation, inceptiontime")

    # Run optimization
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Results
    print(f"\nBest trial:")
    print(f"  Value (Weighted F1): {study.best_trial.value:.4f}")
    print(f"  Params:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")

    # Save results
    completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    results = {
        'model_type': model_type,
        'best_params': study.best_trial.params,
        'best_score': study.best_trial.value,
        'n_trials': completed_trials,  # Actual completed trials (for comparison tracking)
        'timestamp': datetime.now().isoformat()
    }

    with open(output_dir / f'{model_type}_best_params.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_dir / f'{model_type}_best_params.json'}")

    return results


def load_best_params(model_type: str, params_dir: Path = None) -> dict:
    """Load best hyperparameters for a model type."""
    if params_dir is None:
        params_dir = BEST_PARAMS_DIR

    params_file = params_dir / f'{model_type}_best_params.json'
    if not params_file.exists():
        print(f"Warning: No tuned params for {model_type}, using defaults")
        return {}

    with open(params_file) as f:
        data = json.load(f)

    return data.get('best_params', {})


def main():
    ALL_MODELS = ['svm', 'lstm', 'gru', 'transformer', 'tcn',
                  'tsfresh_xgboost', 'lite', 'ts2vec_ft', 'distillation',
                  'inceptiontime']

    parser = argparse.ArgumentParser(description='Optuna hyperparameter tuning')
    parser.add_argument('--model', type=str, required=True,
                        choices=ALL_MODELS + ['all'],
                        help='Model type to tune')
    parser.add_argument('--n-trials', type=int, default=OPTUNA_N_TRIALS,
                        help='Number of Optuna trials')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for results')
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None

    if args.model == 'all':
        for model_type in ALL_MODELS:
            run_optimization(model_type, args.n_trials, output_dir)
    else:
        run_optimization(args.model, args.n_trials, output_dir)


if __name__ == '__main__':
    main()
