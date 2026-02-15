"""
Physics-Informed Data Augmentation for Thermal Time-Series Classification.

================================================================================
OVERVIEW
================================================================================

This module implements augmentations designed specifically for thermal imaging data
based on the physical properties of the acquisition system:
- FLIR T420 thermal camera (NETD ~50mK)
- 2500W heater on conveyor belt
- Material-specific thermal signatures (emissivity, thermal conductivity)

================================================================================
PHYSICAL SYSTEM CONTEXT
================================================================================

| Parameter        | Value           | Implication for Augmentation           |
|------------------|-----------------|----------------------------------------|
| Camera           | FLIR T420       | NETD ~50mK → noise σ=0.05 after z-score|
| Heater           | 2500W           | Fixed heating intensity                |
| Conveyor         | 160×40 cm       | Variable object dwell time             |
| Frame Rate       | 30 FPS          | Fine temporal resolution               |

Material Thermal Signatures (from paper Table 7):
| Material | μ ± σ (°C)   | Key Property           |
|----------|--------------|------------------------|
| Metal    | 33.8 ± 10.0  | High conductivity      |
| Plastic  | 35.7 ± 2.9   | Low conductivity       |
| Glass    | 36.7 ± 1.0   | Very stable            |
| Paper    | 33.0 ± 0.5   | Fast response          |

================================================================================
KEY DESIGN PRINCIPLES
================================================================================

1. **Preserve fall time** - Most discriminative feature (-7.3% accuracy when removed)
   → Use conservative time masking (8% max vs 20%)
   → Use gentle time stretch (±10% max)

2. **Match sensor characteristics** - FLIR T420 NETD ~50mK
   → Noise σ=0.05 after z-score normalization

3. **Avoid destroying temporal structure** - Heating/cooling cycle is essential
   → NO random cropping, NO mixup, NO frequency masking

================================================================================
IMPLEMENTATION PRIORITY
================================================================================

| Priority | Augmentation      | Change                | Risk   | Hypothesis (UNTESTED) |
|----------|-------------------|----------------------|--------|-----------------------|
| **P0**   | Time Masking      | 30%→15%, 20%→8%      | Low    | May help              |
| **P0**   | Noise Injection   | 20%/0.1→100%/0.05    | Low    | May help              |
| **P1**   | Baseline Shift    | None→±0.2, 40%       | Low    | May help              |
| **P1**   | Intensity Scaling | None→[0.92,1.08], 30%| Medium | May help              |
| **P2**   | Time Stretch      | None→[0.9,1.1], 25%  | Medium | May help              |

Note: "Expected Impact" claims were removed. Actual impact requires A/B testing.

================================================================================
USAGE
================================================================================

For Deep Learning Methods (BiLSTM, BiGRU, TCN, Transformer, InceptionTime):
    # Automatic via dataset.py
    from thermal_classifiers.shared.dataset import create_data_loaders
    train_loader, val_loader, test_loader, class_weights = create_data_loaders(
        augment_train=True  # Uses physics-informed augmentation by default
    )

    # Use legacy augmentation for A/B comparison
    train_loader, val_loader, test_loader, class_weights = create_data_loaders(
        augment_train=True,
        use_legacy_augmentation=True
    )

    # Custom configuration
    from thermal_classifiers.shared.augmentation import AugmentationConfig
    custom_config = AugmentationConfig(noise_std=0.03, time_stretch_prob=0.0)
    train_loader, val_loader, test_loader, class_weights = create_data_loaders(
        augment_train=True,
        augmentation_config=custom_config
    )

For Feature-Based Methods (SVM, TSFresh+XGB) and MiniRocket:
    # NOT recommended - these methods are inherently regularized
    # But available if needed:
    from thermal_classifiers.shared.augmentation import create_augmented_dataset
    aug_seqs, aug_labels, aug_ids = create_augmented_dataset(
        sequences, labels, unique_ids,
        n_augmentations=2  # Creates 2 augmented copies per sample
    )

================================================================================
CONFIGURATION PRESETS
================================================================================

DEFAULT_CONFIG: Recommended physics-informed settings
CONSERVATIVE_CONFIG: For methods that are already overfitting
AGGRESSIVE_CONFIG: For testing upper bounds
LEGACY_CONFIG: Matches old dataset.py behavior for comparison

================================================================================
REFERENCE
================================================================================

Full analysis: docs/THERMAL_CLASSIFICATION_FINDINGS.md Section 9
Paper context: recycling_paper_elsevier_article_template/main.tex
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class AugmentationConfig:
    """Configuration for physics-informed thermal augmentation.

    Default values are based on:
    - FLIR T420 specifications (NETD ~50mK → σ=0.05 after z-score)
    - Material thermal variability (Table 7 in paper)
    - SVM ablation study (fall time most important)
    """

    # P0: Time Masking (REDUCED from current)
    time_mask_prob: float = 0.15        # Probability of applying time masking (was 0.30)
    time_mask_max_frac: float = 0.08    # Maximum fraction of sequence to mask (was 0.20)

    # P0: Noise Injection (REDUCED magnitude, INCREASED probability)
    noise_prob: float = 1.0             # Always apply noise (was 0.20)
    noise_std: float = 0.05             # Noise std after z-score (was 0.10)
                                         # Matches FLIR T420 NETD ~50mK

    # P1: Baseline Shift (NEW - models ambient temperature variation)
    baseline_shift_prob: float = 0.40   # Probability of applying baseline shift
    baseline_shift_range: float = 0.2   # ±0.2 shift after z-score normalization
                                         # Represents ~1-2°C ambient variation

    # P1: Intensity Scaling (NEW - models emissivity/angle variation)
    intensity_scale_prob: float = 0.30  # Probability of applying intensity scaling
    intensity_scale_min: float = 0.92   # Minimum scale factor
    intensity_scale_max: float = 1.08   # Maximum scale factor
                                         # Represents ±8% emissivity/angle variation

    # P2: Time Stretch (NEW - models thermal mass variation)
    time_stretch_prob: float = 0.25     # Probability of applying time stretch
    time_stretch_min: float = 0.9       # Minimum stretch factor (1.0 = no change)
    time_stretch_max: float = 1.1       # Maximum stretch factor
                                         # Represents ±10% thermal mass variation


# Default configuration (RECOMMENDED for most methods)
DEFAULT_CONFIG = AugmentationConfig()

# Conservative configuration (for methods that are already overfitting)
CONSERVATIVE_CONFIG = AugmentationConfig(
    time_mask_prob=0.10,
    time_mask_max_frac=0.05,
    noise_prob=1.0,
    noise_std=0.03,
    baseline_shift_prob=0.20,
    baseline_shift_range=0.1,
    intensity_scale_prob=0.15,
    intensity_scale_min=0.95,
    intensity_scale_max=1.05,
    time_stretch_prob=0.0,  # Disable time stretch
)

# Aggressive configuration (for testing upper bounds)
AGGRESSIVE_CONFIG = AugmentationConfig(
    time_mask_prob=0.25,
    time_mask_max_frac=0.12,
    noise_prob=1.0,
    noise_std=0.08,
    baseline_shift_prob=0.50,
    baseline_shift_range=0.3,
    intensity_scale_prob=0.40,
    intensity_scale_min=0.88,
    intensity_scale_max=1.12,
    time_stretch_prob=0.35,
    time_stretch_min=0.85,
    time_stretch_max=1.15,
)

# Legacy configuration (matches current dataset.py for comparison)
LEGACY_CONFIG = AugmentationConfig(
    time_mask_prob=0.30,
    time_mask_max_frac=0.20,
    noise_prob=0.20,
    noise_std=0.10,
    baseline_shift_prob=0.0,
    baseline_shift_range=0.0,
    intensity_scale_prob=0.0,
    intensity_scale_min=1.0,
    intensity_scale_max=1.0,
    time_stretch_prob=0.0,
    time_stretch_min=1.0,
    time_stretch_max=1.0,
)


def apply_time_masking(
    seq: np.ndarray,
    actual_length: int,
    max_len: int,
    prob: float = 0.15,
    max_frac: float = 0.08
) -> np.ndarray:
    """
    Apply time masking augmentation to z-score normalized sequence.

    Replaces a contiguous segment with zeros (mean value after z-score).
    Designed to preserve fall time by limiting mask length.

    Args:
        seq: Sequence array [max_len, 1] (already z-score normalized)
        actual_length: Number of real (non-padded) frames
        max_len: Maximum sequence length
        prob: Probability of applying masking
        max_frac: Maximum fraction of sequence to mask (REDUCED from 0.20)

    Returns:
        Augmented sequence array
    """
    if np.random.random() >= prob:
        return seq

    if actual_length < 10:
        return seq

    # Calculate mask length (limited to max_frac of actual sequence)
    max_mask_length = max(1, int(actual_length * max_frac))
    if max_mask_length <= 1:
        return seq

    mask_length = np.random.randint(1, max_mask_length + 1)

    # Calculate valid positions (after padding at beginning)
    pad_len = max_len - actual_length
    valid_start = pad_len  # First valid index after padding
    valid_end = pad_len + actual_length - mask_length

    if valid_end <= valid_start:
        return seq

    start_idx = np.random.randint(valid_start, valid_end)
    seq[start_idx:start_idx + mask_length] = 0.0  # 0 = mean after z-score

    return seq


def apply_noise_injection(
    seq: np.ndarray,
    actual_length: int,
    max_len: int,
    prob: float = 1.0,
    noise_std: float = 0.05
) -> np.ndarray:
    """
    Apply Gaussian noise injection to z-score normalized sequence.

    Noise magnitude is based on FLIR T420 NETD (~50mK) converted to
    z-score space (roughly 0.05 std after normalization).

    Args:
        seq: Sequence array [max_len, 1] (already z-score normalized)
        actual_length: Number of real (non-padded) frames
        max_len: Maximum sequence length
        prob: Probability of applying noise (default: always)
        noise_std: Noise standard deviation (REDUCED from 0.10)

    Returns:
        Augmented sequence array
    """
    if np.random.random() >= prob:
        return seq

    # Generate noise
    noise = np.random.normal(0, noise_std, seq.shape).astype(np.float32)

    # Only add noise to actual sequence, not padding
    pad_len = max_len - actual_length
    seq[pad_len:] = seq[pad_len:] + noise[pad_len:]

    return seq


def apply_baseline_shift(
    seq: np.ndarray,
    actual_length: int,
    max_len: int,
    prob: float = 0.40,
    shift_range: float = 0.2
) -> np.ndarray:
    """
    Apply baseline shift to model ambient temperature variation.

    Adds a constant offset to the entire sequence, simulating variations
    in ambient temperature (which affects the baseline thermal emission).

    Physical interpretation:
    - Ambient temperature varies by ±1-2°C between experiments
    - After z-score normalization, this corresponds to ±0.2 offset

    Args:
        seq: Sequence array [max_len, 1] (already z-score normalized)
        actual_length: Number of real (non-padded) frames
        max_len: Maximum sequence length
        prob: Probability of applying baseline shift
        shift_range: Maximum shift magnitude (±shift_range)

    Returns:
        Augmented sequence array
    """
    if np.random.random() >= prob:
        return seq

    # Random baseline shift in range [-shift_range, +shift_range]
    shift = np.random.uniform(-shift_range, shift_range)

    # Only shift actual sequence, not padding
    pad_len = max_len - actual_length
    seq[pad_len:] = seq[pad_len:] + shift

    return seq


def apply_intensity_scaling(
    seq: np.ndarray,
    actual_length: int,
    max_len: int,
    prob: float = 0.30,
    scale_min: float = 0.92,
    scale_max: float = 1.08
) -> np.ndarray:
    """
    Apply intensity scaling to model emissivity and viewing angle variation.

    Multiplies the sequence by a scale factor, simulating:
    - Emissivity variation between objects of same material
    - Viewing angle effects on apparent temperature
    - Surface condition variations (oxidation, dirt, etc.)

    Physical interpretation:
    - Emissivity can vary ±5-10% for same material (surface condition)
    - Viewing angle affects apparent temperature by similar margin

    Args:
        seq: Sequence array [max_len, 1] (already z-score normalized)
        actual_length: Number of real (non-padded) frames
        max_len: Maximum sequence length
        prob: Probability of applying intensity scaling
        scale_min: Minimum scale factor
        scale_max: Maximum scale factor

    Returns:
        Augmented sequence array
    """
    if np.random.random() >= prob:
        return seq

    # Random scale factor
    scale = np.random.uniform(scale_min, scale_max)

    # Only scale actual sequence, not padding
    pad_len = max_len - actual_length
    seq[pad_len:] = seq[pad_len:] * scale

    return seq


def apply_time_stretch(
    seq: np.ndarray,
    actual_length: int,
    max_len: int,
    prob: float = 0.25,
    stretch_min: float = 0.9,
    stretch_max: float = 1.1
) -> np.ndarray:
    """
    Apply time stretch to model thermal mass variation.

    Resamples the sequence to simulate objects with different thermal masses
    (heavier/lighter objects heat and cool at different rates).

    CAUTION: This affects fall time, which is the most discriminative feature.
    Keep stretch factors close to 1.0 (±10% max).

    Physical interpretation:
    - Objects of same material but different mass have different thermal dynamics
    - Larger objects heat/cool slower (time stretch > 1)
    - Smaller objects heat/cool faster (time stretch < 1)

    Args:
        seq: Sequence array [max_len, 1] (already z-score normalized)
        actual_length: Number of real (non-padded) frames
        max_len: Maximum sequence length
        prob: Probability of applying time stretch
        stretch_min: Minimum stretch factor (< 1 = compress)
        stretch_max: Maximum stretch factor (> 1 = stretch)

    Returns:
        Augmented sequence array
    """
    if np.random.random() >= prob:
        return seq

    if actual_length < 10:
        return seq

    # Random stretch factor
    stretch = np.random.uniform(stretch_min, stretch_max)

    # Extract actual sequence (excluding padding)
    pad_len = max_len - actual_length
    actual_seq = seq[pad_len:pad_len + actual_length, 0].copy()

    # Calculate new length after stretching
    new_length = int(actual_length * stretch)
    new_length = max(10, min(new_length, max_len))  # Clamp to valid range

    # Resample using linear interpolation
    old_indices = np.linspace(0, actual_length - 1, actual_length)
    new_indices = np.linspace(0, actual_length - 1, new_length)
    stretched_seq = np.interp(new_indices, old_indices, actual_seq)

    # Create new sequence with padding at beginning
    new_pad_len = max_len - new_length
    new_seq = np.zeros((max_len, 1), dtype=np.float32)
    new_seq[new_pad_len:] = stretched_seq.reshape(-1, 1)

    return new_seq


def augment_thermal_sequence(
    seq: np.ndarray,
    actual_length: int,
    max_len: int,
    config: Optional[AugmentationConfig] = None
) -> Tuple[np.ndarray, int]:
    """
    Apply full physics-informed augmentation pipeline to a thermal sequence.

    Applies augmentations in order:
    1. Time stretch (P2) - may change sequence length
    2. Baseline shift (P1)
    3. Intensity scaling (P1)
    4. Time masking (P0)
    5. Noise injection (P0)

    Args:
        seq: Sequence array [max_len, 1] (already z-score normalized)
        actual_length: Number of real (non-padded) frames
        max_len: Maximum sequence length
        config: Augmentation configuration (default: DEFAULT_CONFIG)

    Returns:
        Tuple of (augmented_sequence, new_actual_length)
        - new_actual_length may differ if time stretch was applied
    """
    if config is None:
        config = DEFAULT_CONFIG

    seq = seq.copy()  # Don't modify original
    current_length = actual_length

    # P2: Time stretch (may change length)
    if config.time_stretch_prob > 0:
        old_pad_len = max_len - current_length
        seq = apply_time_stretch(
            seq, current_length, max_len,
            prob=config.time_stretch_prob,
            stretch_min=config.time_stretch_min,
            stretch_max=config.time_stretch_max
        )
        # Recalculate actual length robustly
        # Edge case: first data value could be 0 after z-score (if it equals mean)
        # Solution: find first non-zero index (padding is always 0, data rarely exactly 0)
        nonzero_mask = seq[:, 0] != 0
        if nonzero_mask.any():
            # Find first True in mask (first non-padding position)
            new_pad_len = np.argmax(nonzero_mask)
            # But if first element is non-zero, argmax returns 0 correctly
            if not nonzero_mask[0] and new_pad_len == 0:
                # All zeros case (shouldn't happen)
                new_pad_len = max_len
        else:
            new_pad_len = max_len
        current_length = max_len - new_pad_len

    # P1: Baseline shift
    if config.baseline_shift_prob > 0:
        seq = apply_baseline_shift(
            seq, current_length, max_len,
            prob=config.baseline_shift_prob,
            shift_range=config.baseline_shift_range
        )

    # P1: Intensity scaling
    if config.intensity_scale_prob > 0:
        seq = apply_intensity_scaling(
            seq, current_length, max_len,
            prob=config.intensity_scale_prob,
            scale_min=config.intensity_scale_min,
            scale_max=config.intensity_scale_max
        )

    # P0: Time masking
    if config.time_mask_prob > 0:
        seq = apply_time_masking(
            seq, current_length, max_len,
            prob=config.time_mask_prob,
            max_frac=config.time_mask_max_frac
        )

    # P0: Noise injection (always last to avoid amplifying noise)
    if config.noise_prob > 0:
        seq = apply_noise_injection(
            seq, current_length, max_len,
            prob=config.noise_prob,
            noise_std=config.noise_std
        )

    return seq, current_length


def augment_thermal_sequence_legacy(
    seq: np.ndarray,
    actual_length: int,
    max_len: int
) -> np.ndarray:
    """
    Legacy augmentation matching current dataset.py implementation.

    Use this for baseline comparison.

    Args:
        seq: Sequence array [max_len, 1] (already z-score normalized)
        actual_length: Number of real (non-padded) frames
        max_len: Maximum sequence length

    Returns:
        Augmented sequence array
    """
    seq, _ = augment_thermal_sequence(seq, actual_length, max_len, config=LEGACY_CONFIG)
    return seq


def create_augmented_dataset(
    sequences: list,
    labels: list,
    unique_ids: list,
    n_augmentations: int = 2,
    config: Optional[AugmentationConfig] = None,
    normalize: bool = True
) -> Tuple[list, list, list]:
    """
    Create an augmented dataset for offline training (SVM, MiniRocket, TSFresh+XGB).

    Unlike deep learning where augmentation happens on-the-fly during training,
    feature-based methods require fixed features. This function creates multiple
    augmented copies of each training sample BEFORE feature extraction.

    WARNING: For MiniRocket, augmentation is NOT recommended because random kernels
    are inherently regularized. Use with caution.

    Args:
        sequences: List of 1D numpy arrays (raw thermal intensities)
        labels: List of integer class labels
        unique_ids: List of unique track identifiers
        n_augmentations: Number of augmented copies per sample (default: 2)
        config: Augmentation configuration (default: DEFAULT_CONFIG)
        normalize: Apply z-score normalization before augmentation

    Returns:
        Tuple of (augmented_sequences, augmented_labels, augmented_ids)
        - Original samples are included (total = original * (1 + n_augmentations))
    """
    if config is None:
        config = DEFAULT_CONFIG

    augmented_seqs = []
    augmented_labels = []
    augmented_ids = []

    for seq, label, uid in zip(sequences, labels, unique_ids):
        seq = np.array(seq)

        # Add original sample
        augmented_seqs.append(seq.copy())
        augmented_labels.append(label)
        augmented_ids.append(uid)

        # Z-score normalize for augmentation
        if normalize:
            seq_mean = np.mean(seq)
            seq_std = np.std(seq)
            if seq_std > 1e-8:
                seq_norm = (seq - seq_mean) / seq_std
            else:
                seq_norm = seq - seq_mean
        else:
            seq_norm = seq
            seq_mean = 0
            seq_std = 1

        # Create augmented copies
        max_len = len(seq)
        seq_2d = seq_norm.reshape(-1, 1).astype(np.float32)

        for aug_idx in range(n_augmentations):
            # Apply augmentation
            aug_seq, new_len = augment_thermal_sequence(
                seq_2d.copy(), max_len, max_len, config
            )

            # Denormalize back to original scale
            # CRITICAL: Padding is at BEGINNING, so slice from END
            aug_seq_1d = aug_seq[-new_len:, 0].copy()
            if normalize:
                aug_seq_1d = aug_seq_1d * seq_std + seq_mean

            augmented_seqs.append(aug_seq_1d)
            augmented_labels.append(label)
            augmented_ids.append(f"{uid}_aug{aug_idx}")

    return augmented_seqs, augmented_labels, augmented_ids


if __name__ == '__main__':
    print("Testing physics-informed augmentation module...")

    # Create a test sequence (simulating z-score normalized thermal data)
    np.random.seed(42)
    max_len = 1000
    actual_len = 800

    # Create sequence with heating/cooling pattern
    t = np.linspace(0, 10, actual_len)
    heating = 1 - np.exp(-t/2)  # Exponential heating
    cooling = np.exp(-(t - 5)/3) * (t > 5)  # Exponential cooling after t=5
    raw_seq = heating + cooling
    raw_seq = (raw_seq - raw_seq.mean()) / raw_seq.std()  # Z-score normalize

    # Pad at beginning
    test_seq = np.zeros((max_len, 1), dtype=np.float32)
    test_seq[-actual_len:, 0] = raw_seq

    print(f"Test sequence: shape={test_seq.shape}, actual_len={actual_len}")
    print(f"  Mean: {test_seq[-actual_len:].mean():.4f}, Std: {test_seq[-actual_len:].std():.4f}")

    # Test each augmentation
    print("\n1. Time Masking (P0):")
    masked = apply_time_masking(test_seq.copy(), actual_len, max_len, prob=1.0, max_frac=0.08)
    zeros_count = np.sum(masked[-actual_len:] == 0)
    print(f"  Zeros introduced: {zeros_count} ({100*zeros_count/actual_len:.1f}%)")

    print("\n2. Noise Injection (P0):")
    noisy = apply_noise_injection(test_seq.copy(), actual_len, max_len, prob=1.0, noise_std=0.05)
    noise_diff = np.abs(noisy[-actual_len:] - test_seq[-actual_len:])
    print(f"  Noise magnitude: mean={noise_diff.mean():.4f}, max={noise_diff.max():.4f}")

    print("\n3. Baseline Shift (P1):")
    shifted = apply_baseline_shift(test_seq.copy(), actual_len, max_len, prob=1.0, shift_range=0.2)
    shift_amount = (shifted[-actual_len:].mean() - test_seq[-actual_len:].mean())
    print(f"  Shift amount: {shift_amount:.4f}")

    print("\n4. Intensity Scaling (P1):")
    scaled = apply_intensity_scaling(test_seq.copy(), actual_len, max_len, prob=1.0, scale_min=0.92, scale_max=1.08)
    scale_factor = scaled[-actual_len:].std() / test_seq[-actual_len:].std()
    print(f"  Scale factor: {scale_factor:.4f}")

    print("\n5. Time Stretch (P2):")
    stretched = apply_time_stretch(test_seq.copy(), actual_len, max_len, prob=1.0, stretch_min=0.9, stretch_max=1.1)
    # Recalculate new_len by finding where padding ends
    pad_idx = np.argmax(stretched[:, 0] != 0) if stretched[0, 0] == 0 else 0
    new_len = max_len - pad_idx
    print(f"  Length change: {actual_len} -> {new_len} ({100*(new_len-actual_len)/actual_len:+.1f}%)")

    print("\n6. Full Pipeline (DEFAULT_CONFIG):")
    full_aug, final_len = augment_thermal_sequence(test_seq.copy(), actual_len, max_len, DEFAULT_CONFIG)
    print(f"  Final length: {final_len}")
    print(f"  Final mean: {full_aug[-final_len:].mean():.4f}, std: {full_aug[-final_len:].std():.4f}")

    print("\n7. Legacy Pipeline (comparison):")
    legacy_aug = augment_thermal_sequence_legacy(test_seq.copy(), actual_len, max_len)
    print(f"  Mean: {legacy_aug[-actual_len:].mean():.4f}, std: {legacy_aug[-actual_len:].std():.4f}")

    print("\nAll tests passed!")
