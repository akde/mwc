#!/usr/bin/env python3
"""
Run all thermal classification methods for fair comparison.

This script orchestrates training for all methods:
1. SVM (baseline with 5 statistical features)
2. BiLSTM (bidirectional LSTM)
3. BiGRU (bidirectional GRU)
4. Transformer (encoder with CLS pooling)
5. TCN (temporal convolutional network)
6. 1D CNN (convolutional neural network)
7. InceptionTime (multi-scale inception modules)
8. MiniRocket (random kernel features + Ridge)

Usage:
    python run_all.py
    python run_all.py --skip-tuning
    python run_all.py --methods svm lstm gru
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_DIR))

from thermal_classifiers.shared.config import (
    RESULTS_DIR, OPTUNA_N_TRIALS, OPTUNA_CV_FOLDS, print_config
)

# Methods relevant for the paper (independent, no inter-dependencies)
ALL_METHODS = ['svm', 'lstm', 'gru', 'transformer', 'tcn', 'cnn_1d',
               'inceptiontime', 'minirocket']

# Methods that support Optuna hyperparameter tuning
TUNABLE_METHODS = {'svm', 'lstm', 'gru', 'transformer', 'tcn', 'cnn_1d',
                   'inceptiontime'}

# DL methods that use the unified train_dl.py script
DL_METHODS = {'lstm', 'gru', 'transformer', 'cnn_1d', 'tcn', 'inceptiontime'}

# Method display names
METHOD_NAMES = {
    'svm': 'SVM (5 Features)',
    'lstm': 'BiLSTM',
    'gru': 'BiGRU',
    'transformer': 'Transformer',
    'tcn': 'TCN',
    'cnn_1d': '1D CNN',
    'inceptiontime': 'InceptionTime',
    'minirocket': 'MiniRocket',
}


def run_method(method: str, skip_tuning: bool, n_trials: int) -> Tuple[bool, float]:
    """
    Run training for a single method.

    Returns:
        (success, elapsed_time_seconds)
    """
    if method in DL_METHODS:
        # Use unified DL training script
        cmd = [sys.executable, str(CODE_DIR / 'thermal_classifiers' / 'train_dl.py'),
               '--method', method]
    else:
        # Method-specific training script
        cmd = [sys.executable, '-m', f'thermal_classifiers.{method}.train']

    if method in TUNABLE_METHODS:
        if skip_tuning:
            cmd.append('--skip-tuning')
        else:
            cmd.extend(['--n-trials', str(n_trials)])

    env = os.environ.copy()
    env['MPLBACKEND'] = 'Agg'  # Non-interactive matplotlib backend

    start_time = time.time()
    try:
        result = subprocess.run(cmd, cwd=str(CODE_DIR), env=env, check=True)
        return True, time.time() - start_time
    except subprocess.CalledProcessError as e:
        print(f"Error running {method}: {e}")
        return False, time.time() - start_time


def format_time(seconds: float) -> str:
    """Format seconds as human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds // 60:.0f}m {seconds % 60:.0f}s"
    else:
        return f"{seconds // 3600:.0f}h {(seconds % 3600) // 60:.0f}m"


def main():
    parser = argparse.ArgumentParser(
        description='Run all thermal classification methods'
    )
    parser.add_argument('--methods', nargs='+', default=ALL_METHODS,
                        choices=ALL_METHODS,
                        help='Methods to run (default: all)')
    parser.add_argument('--skip-tuning', action='store_true',
                        help='Skip hyperparameter tuning, use defaults')
    parser.add_argument('--n-trials', type=int, default=OPTUNA_N_TRIALS,
                        help='Number of Optuna trials per method')
    args = parser.parse_args()

    print("=" * 70)
    print("THERMAL CLASSIFICATION - RUN ALL METHODS")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Methods: {args.methods}")
    print(f"Skip tuning: {args.skip_tuning}")
    if not args.skip_tuning:
        print(f"Optuna trials: {args.n_trials}")
    print(f"Results directory: {RESULTS_DIR}")
    print("=" * 70)

    results = {}
    total_start = time.time()

    for i, method in enumerate(args.methods, 1):
        print(f"\n[{i}/{len(args.methods)}] Running {METHOD_NAMES.get(method, method)}...")
        print("-" * 40)

        success, elapsed = run_method(method, args.skip_tuning, args.n_trials)
        results[method] = {'success': success, 'time': elapsed}
        status = "DONE" if success else "FAILED"
        print(f"\n[{i}/{len(args.methods)}] {METHOD_NAMES.get(method, method)}: {status} ({format_time(elapsed)})")

    total_time = time.time() - total_start

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    successes = sum(1 for r in results.values() if r['success'])
    failures = len(results) - successes

    print(f"\nResults: {successes}/{len(results)} succeeded, {failures} failed")

    if failures > 0:
        failed = [m for m, r in results.items() if not r['success']]
        print(f"Failed methods: {failed}")

    print(f"\nTime per method:")
    for method, r in results.items():
        status = "OK" if r['success'] else "FAIL"
        print(f"  [{status}] {METHOD_NAMES.get(method, method)}: {format_time(r['time'])}")

    print(f"\nTotal time: {format_time(total_time)}")
    print("=" * 70)

    return 1 if failures > 0 else 0


if __name__ == '__main__':
    sys.exit(main())
