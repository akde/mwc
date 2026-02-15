#!/usr/bin/env python3
"""
Create stratified 3-way train/val/test splits for both RGB and thermal classification.

Updates labels.csv with train/val/test split assignments using stratified sampling
to maintain class proportions across all splits.

Split ratio: 60% train / 20% val / 20% test (stratified by class)

Usage:
    python create_splits.py --labels-csv ../../labels.csv
    python create_splits.py --labels-csv ../../labels.csv --train-ratio 0.7 --val-ratio 0.15
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path


def create_stratified_splits(
    labels_csv: str,
    train_ratio: float = 0.60,
    val_ratio: float = 0.20,
    test_ratio: float = 0.20,
    random_seed: int = 42,
    dry_run: bool = False
):
    """Create stratified 3-way splits and update labels.csv."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"

    df = pd.read_csv(labels_csv)
    print(f"Loaded {len(df)} tracklets from {labels_csv}")
    print(f"\nClass distribution:")
    print(df['class'].value_counts().to_string())
    print(f"\nTarget split: {train_ratio:.0%} train / {val_ratio:.0%} val / {test_ratio:.0%} test")

    # First split: train vs (val+test)
    val_test_ratio = val_ratio + test_ratio
    train_idx, valtest_idx = train_test_split(
        df.index, test_size=val_test_ratio, random_state=random_seed,
        stratify=df['class']
    )

    # Second split: val vs test (from the val+test portion)
    relative_test_ratio = test_ratio / val_test_ratio
    val_idx, test_idx = train_test_split(
        valtest_idx, test_size=relative_test_ratio, random_state=random_seed,
        stratify=df.loc[valtest_idx, 'class']
    )

    # Assign splits
    df['split'] = ''
    df.loc[train_idx, 'split'] = 'train'
    df.loc[val_idx, 'split'] = 'val'
    df.loc[test_idx, 'split'] = 'test'

    # Verify
    print(f"\nActual split:")
    print(df['split'].value_counts().to_string())
    print(f"\nClass x Split:")
    ct = pd.crosstab(df['class'], df['split'])
    print(ct.to_string())

    # Check proportions per class
    print(f"\nProportion per class:")
    for cls in sorted(df['class'].unique()):
        cls_df = df[df['class'] == cls]
        n = len(cls_df)
        n_train = len(cls_df[cls_df['split'] == 'train'])
        n_val = len(cls_df[cls_df['split'] == 'val'])
        n_test = len(cls_df[cls_df['split'] == 'test'])
        print(f"  {cls:8s}: {n_train}/{n_val}/{n_test} "
              f"({n_train/n:.1%}/{n_val/n:.1%}/{n_test/n:.1%})")

    if not dry_run:
        df.to_csv(labels_csv, index=False)
        print(f"\nSaved updated labels to {labels_csv}")
    else:
        print(f"\n[DRY RUN] Would save to {labels_csv}")

    return df


def main():
    parser = argparse.ArgumentParser(description='Create stratified 3-way splits')
    parser.add_argument('--labels-csv', type=str,
                        default=str(Path(__file__).resolve().parent.parent / 'labels.csv'),
                        help='Path to labels.csv')
    parser.add_argument('--train-ratio', type=float, default=0.60)
    parser.add_argument('--val-ratio', type=float, default=0.20)
    parser.add_argument('--test-ratio', type=float, default=0.20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dry-run', action='store_true',
                        help='Print splits without saving')
    args = parser.parse_args()

    create_stratified_splits(
        args.labels_csv,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed,
        dry_run=args.dry_run
    )


if __name__ == '__main__':
    main()
