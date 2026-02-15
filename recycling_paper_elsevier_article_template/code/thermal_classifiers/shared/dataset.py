"""
Dataset utilities for thermal time series classification.

This module provides:
- ThermalSequenceDataset: PyTorch Dataset for variable-length thermal intensity sequences
- load_data: Load and parse CSV data
- create_data_loaders: Create train/val/test DataLoaders with proper settings
- compute_class_weights: Calculate balanced class weights for imbalanced data

Key difference from RGB classification:
- Input is univariate (1 channel) thermal intensity, not 4-channel probability vectors
- Per-track z-score normalization applied for better model convergence
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from collections import Counter

from .config import (
    CLASS_NAMES, CLASS_TO_IDX, IDX_TO_CLASS, NUM_CLASSES,
    MAX_SEQ_LEN, SUBSAMPLE_STRIDE, INPUT_SIZE,
    BATCH_SIZE, RANDOM_SEED,
    TRAIN_DATA_PATH, VAL_DATA_PATH, TEST_DATA_PATH,
    USE_LEGACY_AUGMENTATION
)
from .augmentation import (
    augment_thermal_sequence,
    augment_thermal_sequence_legacy,
    AugmentationConfig,
    DEFAULT_CONFIG,
    LEGACY_CONFIG
)


class ThermalSequenceDataset(Dataset):
    """
    PyTorch Dataset for univariate thermal time series.

    Each sample is a sequence of scalar thermal intensity values.

    Features:
    - Per-track z-score normalization: (x - mean) / std
    - Variable-length sequence handling via subsampling/truncation/padding
    - Data augmentation (time masking + noise injection) for training
    - Returns actual sequence length and attention mask for proper masking in models

    CSV Format: class, uniqueID, track_length, mean_thermal_intensity_array
    """

    def __init__(
        self,
        sequences: List[np.ndarray],
        labels: List[int],
        unique_ids: List[str],
        max_len: int = MAX_SEQ_LEN,
        subsample: int = SUBSAMPLE_STRIDE,
        normalize: bool = True,
        augment: bool = False,
        augmentation_config: Optional[AugmentationConfig] = None,
        use_legacy_augmentation: bool = False,
        use_interpolation_resize: bool = True  # FAIRNESS: Match RGB preprocessing
    ):
        """
        Initialize dataset.

        Args:
            sequences: List of 1D numpy arrays (variable length thermal intensities)
            labels: List of integer class labels
            unique_ids: List of unique track identifiers
            max_len: Maximum sequence length (resize/pad/truncate to this)
            subsample: Subsample stride for sequences > max_len * subsample
            normalize: Apply per-track z-score normalization
            augment: Apply data augmentation (training only)
            augmentation_config: Physics-informed augmentation config (default: DEFAULT_CONFIG)
            use_legacy_augmentation: If True, use legacy augmentation for comparison
            use_interpolation_resize: If True, use interpolation to resize sequences to
                                     exact max_len (no padding/truncation). Matches RGB.
        """
        assert len(sequences) == len(labels) == len(unique_ids), \
            "Sequences, labels, and unique_ids must have same length"

        self.sequences = sequences
        self.labels = labels
        self.unique_ids = unique_ids
        self.max_len = max_len
        self.subsample = subsample
        self.normalize = normalize
        self.augment = augment
        self.use_legacy_augmentation = use_legacy_augmentation
        self.augmentation_config = augmentation_config if augmentation_config else DEFAULT_CONFIG
        self.use_interpolation_resize = use_interpolation_resize

    def __len__(self) -> int:
        return len(self.sequences)

    def _resize_sequence(self, seq: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Resize sequence to max_len via linear interpolation.

        FAIRNESS: Matches RGB's _resize_sequence. Instead of truncate/pad which
        loses information or adds artificial zeros, we resample the entire
        sequence to target length, preserving temporal patterns.

        Args:
            seq: Input sequence of shape [seq_len] (1D thermal intensity)

        Returns:
            Tuple of (resized_sequence, attention_mask, actual_length)
            - resized_sequence: [max_len] array
            - attention_mask: all 1s (no padding)
            - actual_length: max_len (all positions are valid)
        """
        seq_len = len(seq)
        target_len = self.max_len

        if seq_len == target_len:
            mask = np.ones(target_len, dtype=np.float32)
            return seq, mask, target_len

        # Linear interpolation along time axis
        x_old = np.linspace(0, 1, seq_len)
        x_new = np.linspace(0, 1, target_len)
        resized = np.interp(x_new, x_old, seq).astype(np.float32)

        # All positions are valid (no padding)
        mask = np.ones(target_len, dtype=np.float32)

        return resized, mask, target_len

    def _augment_sequence(self, seq: np.ndarray, actual_length: int) -> Tuple[np.ndarray, int]:
        """
        Apply physics-informed data augmentation for thermal time-series.

        Uses the augmentation module (shared/augmentation.py) which implements:
        - P0: Reduced time masking (15% vs 30%), reduced noise (σ=0.05 vs 0.10)
        - P1: Baseline shift (±0.2), intensity scaling ([0.92, 1.08])
        - P2: Time stretch ([0.9, 1.1])

        For legacy comparison, set use_legacy_augmentation=True in __init__.

        Args:
            seq: Sequence array (already z-score normalized) [max_len, 1]
            actual_length: Number of real (non-padded) frames

        Returns:
            Tuple of (augmented sequence, new actual length)
            - new actual length may differ if time stretch was applied
        """
        if not self.augment:
            return seq, actual_length

        if self.use_legacy_augmentation:
            # Use legacy augmentation for backward compatibility/comparison
            augmented = augment_thermal_sequence_legacy(seq, actual_length, self.max_len)
            return augmented, actual_length
        else:
            # Use new physics-informed augmentation
            return augment_thermal_sequence(
                seq, actual_length, self.max_len,
                config=self.augmentation_config
            )

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
        """
        Get a single sample.

        Returns:
            Tuple of:
                - sequence: [max_len, 1] tensor (normalized, padded/truncated)
                - label: scalar tensor (class index)
                - length: scalar tensor (actual sequence length BEFORE padding)
                - mask: [max_len] tensor (1 for real data, 0 for padding)
                - unique_id: string identifier
        """
        seq = self.sequences[idx].copy()  # 1D array
        label = self.labels[idx]
        unique_id = self.unique_ids[idx]

        # Per-track z-score normalization
        if self.normalize:
            seq_mean = np.mean(seq)
            seq_std = np.std(seq)
            if seq_std > 1e-8:  # Avoid division by zero
                seq = (seq - seq_mean) / seq_std
            else:
                seq = seq - seq_mean

        # Subsample if very long (> max_len * subsample)
        if len(seq) > self.max_len * self.subsample:
            seq = seq[::self.subsample]

        # FAIRNESS: Use interpolation resize (matches RGB) or legacy truncate/pad
        if self.use_interpolation_resize:
            # Resize via linear interpolation - preserves full temporal pattern
            seq, mask, original_len = self._resize_sequence(seq)
        else:
            # Legacy behavior: truncate/pad
            # Track original length (before truncation/padding)
            original_len = min(len(seq), self.max_len)

            # Truncate if still too long (keep MOST RECENT frames)
            if len(seq) > self.max_len:
                seq = seq[-self.max_len:]

            # Pad at BEGINNING (aligned with RGB) so most recent frames at end
            if len(seq) < self.max_len:
                pad_len = self.max_len - len(seq)
                seq = np.pad(seq, (pad_len, 0), mode='constant', constant_values=0)
                # Create mask: 0 for padding, 1 for real data
                mask = np.concatenate([
                    np.zeros(pad_len, dtype=np.float32),
                    np.ones(original_len, dtype=np.float32)
                ])
            else:
                mask = np.ones(self.max_len, dtype=np.float32)

        # Reshape to [max_len, 1] for model compatibility
        seq = seq.reshape(-1, 1).astype(np.float32)

        # Apply augmentation (after normalization and padding)
        # Note: augmentation may change sequence length (time stretch)
        seq, augmented_len = self._augment_sequence(seq, original_len)

        # Update mask if length changed due to time stretch
        if augmented_len != original_len:
            new_pad_len = self.max_len - augmented_len
            mask = np.concatenate([
                np.zeros(new_pad_len, dtype=np.float32),
                np.ones(augmented_len, dtype=np.float32)
            ])
            original_len = augmented_len

        # Convert to tensors
        sequence_tensor = torch.tensor(seq, dtype=torch.float32)  # [max_len, 1]
        label_tensor = torch.tensor(label, dtype=torch.long)
        length_tensor = torch.tensor(original_len, dtype=torch.long)
        mask_tensor = torch.tensor(mask, dtype=torch.float32)  # [max_len]

        return sequence_tensor, label_tensor, length_tensor, mask_tensor, unique_id

    def get_class_counts(self) -> Dict[str, int]:
        """Get count of samples per class."""
        idx_counts = Counter(self.labels)
        return {CLASS_NAMES[idx]: count for idx, count in idx_counts.items()}

    def get_sequence_length_stats(self) -> Dict[str, float]:
        """Get statistics about sequence lengths."""
        lengths = [len(seq) for seq in self.sequences]
        return {
            'min': min(lengths),
            'max': max(lengths),
            'mean': np.mean(lengths),
            'median': np.median(lengths),
            'std': np.std(lengths)
        }

    def get_raw_sequences_for_svm(self) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Get raw (unnormalized) sequences for SVM feature extraction.

        Returns:
            Tuple of (sequences, labels)
        """
        return self.sequences, np.array(self.labels)


def load_data(csv_path: str, verbose: bool = True) -> Tuple[List[np.ndarray], List[int], List[str]]:
    """
    Load and parse thermal sequence data from CSV.

    CSV Format: class, uniqueID, track_length, mean_thermal_intensity_array
    The mean_thermal_intensity_array column contains space-separated float values.

    Args:
        csv_path: Path to CSV file with thermal intensity arrays
        verbose: Print loading info

    Returns:
        Tuple of (sequences, labels, unique_ids)
        - sequences: List of 1D numpy arrays (raw thermal intensities)
        - labels: List of integer class labels
        - unique_ids: List of track identifiers
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if verbose:
        print(f"Loading data from: {csv_path}")
        print(f"  Samples: {len(df)}")

    sequences = []
    labels = []
    unique_ids = []

    for idx, row in df.iterrows():
        # Parse thermal intensity array from space-separated string
        thermal_array = np.array([float(v) for v in str(row['mean_thermal_intensity_array']).split()])

        sequences.append(thermal_array)
        labels.append(CLASS_TO_IDX[row['class']])
        unique_ids.append(str(row['uniqueID']))

    if verbose:
        print(f"  Classes: {dict(pd.Series([CLASS_NAMES[l] for l in labels]).value_counts())}")
        lengths = [len(s) for s in sequences]
        print(f"  Sequence lengths: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.0f}")

    return sequences, labels, unique_ids


def compute_class_weights_tensor(labels: List[int], device: torch.device = None) -> torch.Tensor:
    """
    Compute balanced class weights for handling class imbalance.

    Args:
        labels: List of integer class labels
        device: Torch device for the output tensor

    Returns:
        Tensor of class weights [NUM_CLASSES]
    """
    classes = np.unique(labels)
    weights = compute_class_weight('balanced', classes=classes, y=labels)

    # Create full weight tensor (in case some classes are missing)
    full_weights = np.ones(NUM_CLASSES)
    for cls, w in zip(classes, weights):
        full_weights[cls] = w

    weight_tensor = torch.tensor(full_weights, dtype=torch.float32)
    if device is not None:
        weight_tensor = weight_tensor.to(device)

    return weight_tensor


def collate_fn(batch: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    """
    Custom collate function for DataLoader.

    Handles the 5-tuple return format from ThermalSequenceDataset:
    (sequence, label, length, mask, unique_id)

    Returns:
        Tuple of:
            - sequences: [batch, max_len, 1]
            - labels: [batch]
            - lengths: [batch]
            - masks: [batch, max_len]
            - unique_ids: List[str]
    """
    sequences = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    lengths = torch.stack([item[2] for item in batch])
    masks = torch.stack([item[3] for item in batch])
    unique_ids = [item[4] for item in batch]

    return sequences, labels, lengths, masks, unique_ids


def create_data_loaders(
    train_path: str = None,
    val_path: str = None,
    test_path: str = None,
    batch_size: int = BATCH_SIZE,
    max_len: int = MAX_SEQ_LEN,
    num_workers: int = 0,
    normalize: bool = True,
    augment_train: bool = True,
    augmentation_config: Optional[AugmentationConfig] = None,
    use_legacy_augmentation: bool = None,  # None = read from config.USE_LEGACY_AUGMENTATION
    use_interpolation_resize: bool = True,  # FAIRNESS: Match RGB preprocessing
    verbose: bool = True
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], torch.Tensor]:
    """
    Create train, validation, and test DataLoaders.

    Args:
        train_path: Path to training CSV (default: TRAIN_DATA_PATH)
        val_path: Path to validation CSV (default: VAL_DATA_PATH)
        test_path: Path to test CSV (default: TEST_DATA_PATH)
        batch_size: Batch size for DataLoaders
        max_len: Maximum sequence length
        num_workers: Number of worker processes for data loading
        normalize: Apply per-track z-score normalization
        augment_train: Apply data augmentation to training data
        augmentation_config: Physics-informed augmentation config (default: DEFAULT_CONFIG)
        use_legacy_augmentation: If True, use legacy augmentation for comparison
        use_interpolation_resize: If True, use interpolation to resize sequences (matches RGB)
        verbose: Print loading info

    Returns:
        Tuple of (train_loader, val_loader, test_loader, class_weights)
    """
    train_path = train_path or str(TRAIN_DATA_PATH)
    val_path = val_path or str(VAL_DATA_PATH)
    test_path = test_path or str(TEST_DATA_PATH)

    # Default to config value if not explicitly specified
    if use_legacy_augmentation is None:
        use_legacy_augmentation = USE_LEGACY_AUGMENTATION

    # Load data
    train_seqs, train_labels, train_ids = load_data(train_path, verbose=verbose)
    val_seqs, val_labels, val_ids = load_data(val_path, verbose=verbose)

    test_loader = None
    if Path(test_path).exists():
        test_seqs, test_labels, test_ids = load_data(test_path, verbose=verbose)
        test_dataset = ThermalSequenceDataset(
            test_seqs, test_labels, test_ids, max_len=max_len, normalize=normalize,
            augment=False,  # Never augment test data
            use_interpolation_resize=use_interpolation_resize
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn
        )

    # Create datasets
    train_dataset = ThermalSequenceDataset(
        train_seqs, train_labels, train_ids, max_len=max_len, normalize=normalize,
        augment=augment_train,  # Augment training data
        augmentation_config=augmentation_config,
        use_legacy_augmentation=use_legacy_augmentation,
        use_interpolation_resize=use_interpolation_resize
    )
    val_dataset = ThermalSequenceDataset(
        val_seqs, val_labels, val_ids, max_len=max_len, normalize=normalize,
        augment=False,  # Never augment validation data
        use_interpolation_resize=use_interpolation_resize
    )

    # Compute class weights from training data
    class_weights = compute_class_weights_tensor(train_labels)

    # Create DataLoaders with reproducible shuffling
    g = torch.Generator()
    g.manual_seed(RANDOM_SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=g,
        drop_last=False,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    if verbose:
        print(f"\nDataLoaders created:")
        print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
        print(f"  Val: {len(val_dataset)} samples, {len(val_loader)} batches")
        if test_loader:
            print(f"  Test: {len(test_dataset)} samples, {len(test_loader)} batches")
        aug_type = "legacy" if use_legacy_augmentation else "physics-informed"
        print(f"  Augmentation: train={augment_train} ({aug_type}), val=False, test=False")
        if augment_train and not use_legacy_augmentation:
            cfg = augmentation_config or DEFAULT_CONFIG
            print(f"    Time mask: {cfg.time_mask_prob*100:.0f}% prob, {cfg.time_mask_max_frac*100:.0f}% max len")
            print(f"    Noise: {cfg.noise_prob*100:.0f}% prob, σ={cfg.noise_std}")
            print(f"    Baseline shift: {cfg.baseline_shift_prob*100:.0f}% prob, ±{cfg.baseline_shift_range}")
            print(f"    Intensity scale: {cfg.intensity_scale_prob*100:.0f}% prob, [{cfg.intensity_scale_min:.2f}, {cfg.intensity_scale_max:.2f}]")
            print(f"    Time stretch: {cfg.time_stretch_prob*100:.0f}% prob, [{cfg.time_stretch_min:.2f}, {cfg.time_stretch_max:.2f}]")
        preprocess_mode = "interpolation resize" if use_interpolation_resize else "truncate/pad"
        print(f"  Sequence preprocessing: {preprocess_mode} (FAIRNESS: matches RGB)")
        print(f"  Class weights: {dict(zip(CLASS_NAMES, class_weights.tolist()))}")

    return train_loader, val_loader, test_loader, class_weights


def get_datasets(
    train_path: str = None,
    val_path: str = None,
    test_path: str = None,
    max_len: int = MAX_SEQ_LEN,
    normalize: bool = True,
    verbose: bool = True
) -> Tuple[ThermalSequenceDataset, ThermalSequenceDataset, Optional[ThermalSequenceDataset]]:
    """
    Get Dataset objects directly (useful for SVM feature extraction).

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    train_path = train_path or str(TRAIN_DATA_PATH)
    val_path = val_path or str(VAL_DATA_PATH)
    test_path = test_path or str(TEST_DATA_PATH)

    train_seqs, train_labels, train_ids = load_data(train_path, verbose=verbose)
    val_seqs, val_labels, val_ids = load_data(val_path, verbose=verbose)

    train_dataset = ThermalSequenceDataset(
        train_seqs, train_labels, train_ids, max_len=max_len, normalize=normalize
    )
    val_dataset = ThermalSequenceDataset(
        val_seqs, val_labels, val_ids, max_len=max_len, normalize=normalize
    )

    test_dataset = None
    if Path(test_path).exists():
        test_seqs, test_labels, test_ids = load_data(test_path, verbose=verbose)
        test_dataset = ThermalSequenceDataset(
            test_seqs, test_labels, test_ids, max_len=max_len, normalize=normalize
        )

    return train_dataset, val_dataset, test_dataset


if __name__ == '__main__':
    print("Testing ThermalSequenceDataset...")

    # Load data
    train_loader, val_loader, test_loader, class_weights = create_data_loaders(verbose=True)

    # Test iteration
    print("\nTesting data loading:")
    for batch_idx, (sequences, labels, lengths, masks, unique_ids) in enumerate(train_loader):
        print(f"  Batch {batch_idx}:")
        print(f"    Sequences shape: {sequences.shape}")  # [batch, max_len, 1]
        print(f"    Labels shape: {labels.shape}")
        print(f"    Lengths: {lengths.tolist()[:5]}...")
        print(f"    Masks shape: {masks.shape}")  # [batch, max_len]
        print(f"    Unique IDs: {unique_ids[:3]}...")

        # Check normalized values and mask alignment
        for i in range(min(3, len(sequences))):
            length = lengths[i].item()
            mask = masks[i]
            # With beginning padding, valid positions are at the END
            valid_indices = torch.where(mask > 0)[0]
            if len(valid_indices) > 0:
                seq = sequences[i, valid_indices, 0]
                print(f"    Sample {i} - length: {length}, valid_count: {mask.sum().item():.0f}, "
                      f"mean: {seq.mean():.3f}, std: {seq.std():.3f}")

                # Verify mask is correct (padding at beginning)
                expected_pad_len = masks.shape[1] - length
                if expected_pad_len > 0:
                    assert mask[:expected_pad_len].sum() == 0, "Padding should be zeros at beginning"
                    assert mask[expected_pad_len:].sum() == length, "Real data should be ones at end"

        if batch_idx >= 1:
            break

    print("\nAll tests passed!")
