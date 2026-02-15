"""
Ensemble methods for thermal classification using soft voting.

Combines predictions from TCN, BiLSTM, and 1D CNN models.

Architecture Loading Strategy (December 2025):
    This module supports dynamic architecture loading from hyperparameters.json,
    enabling fair comparison with RGB classifiers by using models trained in
    the current run rather than pre-existing ablation study models.

    Priority order for model loading:
    1. Standard training: results/{model}/model.pt + hyperparameters.json
    2. Standard checkpoint: results/{model}/checkpoints/{model}_best.pt
    3. Ablation studies: results/{model}_ablation/.../
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json

import sys
CODE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(CODE_DIR))

from thermal_classifiers.shared.config import (
    NUM_CLASSES, INPUT_SIZE, RESULTS_DIR, CLASS_NAMES
)


# =============================================================================
# Model Path Resolution
# =============================================================================

def resolve_model_paths() -> Dict[str, Dict[str, Path]]:
    """
    Resolve paths for all ensemble components with priority-based fallback.

    Standard training outputs are preferred over ablation studies for fair
    comparison with RGB classifiers.

    Returns:
        Dict mapping model name to {'weights': Path, 'hyperparams': Path}

    Raises:
        FileNotFoundError: If required model files cannot be found

    Priority order for each model:
        1. Standard training: results/{model}/model.pt + hyperparameters.json
        2. Ablation studies (for backward compatibility)
    """
    paths = {}

    # TCN path resolution
    tcn_candidates = [
        # Standard training outputs (preferred)
        {
            'weights': RESULTS_DIR / 'tcn' / 'model.pt',
            'hyperparams': RESULTS_DIR / 'tcn' / 'hyperparameters.json'
        },
        {
            'weights': RESULTS_DIR / 'tcn' / 'checkpoints' / 'tcn_best.pt',
            'hyperparams': RESULTS_DIR / 'tcn' / 'hyperparameters.json'
        },
        # Augmentation ablation (high-performing)
        {
            'weights': RESULTS_DIR / 'augmentation_ablation' / 'tcn_default' / 'model.pt',
            'hyperparams': None  # Use hardcoded architecture
        },
        # TCN ablation
        {
            'weights': RESULTS_DIR / 'tcn_ablation' / '09_128x1_k3_last_drop03' / 'checkpoints' / 'tcn_best.pt',
            'hyperparams': None  # Use hardcoded architecture
        }
    ]
    paths['tcn'] = _find_first_existing(tcn_candidates, 'TCN')

    # LSTM path resolution
    lstm_candidates = [
        # Standard training outputs (preferred)
        {
            'weights': RESULTS_DIR / 'lstm' / 'model.pt',
            'hyperparams': RESULTS_DIR / 'lstm' / 'hyperparameters.json'
        },
        {
            'weights': RESULTS_DIR / 'lstm' / 'checkpoints' / 'bilstm_best.pt',
            'hyperparams': RESULTS_DIR / 'lstm' / 'hyperparameters.json'
        },
        # LSTM ablation
        {
            'weights': RESULTS_DIR / 'lstm_ablation' / '09_legacy_hidden128' / 'checkpoints' / 'bilstm_best.pt',
            'hyperparams': None  # Use hardcoded architecture
        }
    ]
    paths['lstm'] = _find_first_existing(lstm_candidates, 'LSTM')

    # CNN path resolution
    cnn_candidates = [
        # Standard training outputs (only option)
        {
            'weights': RESULTS_DIR / 'cnn_1d' / 'model.pt',
            'hyperparams': RESULTS_DIR / 'cnn_1d' / 'hyperparameters.json'
        },
        {
            'weights': RESULTS_DIR / 'cnn_1d' / 'checkpoints' / 'cnn_1d_best.pt',
            'hyperparams': RESULTS_DIR / 'cnn_1d' / 'hyperparameters.json'
        }
    ]
    paths['cnn'] = _find_first_existing(cnn_candidates, 'CNN')

    return paths


def _find_first_existing(
    candidates: List[Dict[str, Optional[Path]]],
    model_name: str
) -> Dict[str, Path]:
    """
    Find first existing candidate path set.

    Args:
        candidates: List of {'weights': Path, 'hyperparams': Path or None}
        model_name: Name for error messages

    Returns:
        {'weights': Path, 'hyperparams': Path or None}

    Raises:
        FileNotFoundError: If no candidate exists
    """
    for candidate in candidates:
        weights_path = candidate['weights']
        if weights_path.exists():
            hyperparams_path = candidate.get('hyperparams')
            # Verify hyperparams exists if specified
            if hyperparams_path is not None and not hyperparams_path.exists():
                continue  # Skip this candidate
            return candidate

    # No candidates found
    tried_paths = [str(c['weights']) for c in candidates]
    raise FileNotFoundError(
        f"No {model_name} model found. Tried:\n  " + "\n  ".join(tried_paths)
    )


# =============================================================================
# Dynamic Architecture Creation
# =============================================================================

def create_model_from_hyperparams(
    model_type: str,
    hyperparams: Dict[str, Any],
    device: torch.device
) -> nn.Module:
    """
    Create model with architecture matching hyperparameters.json.

    Args:
        model_type: 'tcn', 'lstm', or 'cnn'
        hyperparams: Dictionary from hyperparameters.json
        device: Target device

    Returns:
        Initialized model (weights not loaded)

    Raises:
        ValueError: If model_type not recognized
    """
    if model_type == 'tcn':
        from thermal_classifiers.tcn.model import create_tcn_model
        return create_tcn_model(
            input_size=INPUT_SIZE,
            num_channels=hyperparams['num_channels'],
            kernel_size=hyperparams['kernel_size'],
            num_classes=NUM_CLASSES,
            dropout=hyperparams['dropout'],
            pooling=hyperparams['pooling']
        )

    elif model_type == 'lstm':
        from thermal_classifiers.lstm.model import create_lstm_model
        return create_lstm_model(
            input_size=INPUT_SIZE,
            hidden_dim=hyperparams['hidden_dim'],
            num_layers=hyperparams['num_layers'],
            num_classes=NUM_CLASSES,
            dropout=hyperparams['dropout'],
            bidirectional=hyperparams.get('bidirectional', True),
            pooling=hyperparams['pooling'],
            use_input_projection=hyperparams.get('use_input_projection', False),
            use_deep_classifier=hyperparams.get('use_deep_classifier', False)
        )

    elif model_type == 'cnn':
        from thermal_classifiers.cnn_1d.model import create_cnn_1d_model
        return create_cnn_1d_model(
            input_size=INPUT_SIZE,
            num_filters=hyperparams['num_filters'],
            kernel_sizes=hyperparams['kernel_sizes'],
            num_classes=NUM_CLASSES,
            dropout=hyperparams['dropout'],
            pooling=hyperparams['pooling'],
            use_batch_norm=hyperparams.get('use_batch_norm', True),
            pool_stride=hyperparams.get('pool_stride', 2)
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_hardcoded_hyperparams(model_type: str) -> Dict[str, Any]:
    """
    Get hardcoded hyperparameters for ablation models (backward compatibility).

    These match the ablation study configurations that ensemble originally used.

    Args:
        model_type: 'tcn', 'lstm', or 'cnn'

    Returns:
        Hyperparameters dictionary
    """
    if model_type == 'tcn':
        # Ablation: 09_128x1_k3_last_drop03
        return {
            'num_channels': [128],
            'kernel_size': 3,
            'dropout': 0.3,
            'pooling': 'last'
        }
    elif model_type == 'lstm':
        # Ablation: 09_legacy_hidden128
        return {
            'hidden_dim': 128,
            'num_layers': 2,
            'dropout': 0.3,
            'bidirectional': True,
            'pooling': 'last',
            'use_input_projection': False,
            'use_deep_classifier': False
        }
    elif model_type == 'cnn':
        # Standard cnn_1d config
        return {
            'num_filters': [64, 128],
            'kernel_sizes': [9, 7],
            'dropout': 0.3,
            'pooling': 'last',
            'use_batch_norm': True,
            'pool_stride': 2
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def load_model_with_architecture(
    model_type: str,
    weights_path: Path,
    hyperparams_path: Optional[Path],
    device: torch.device
) -> nn.Module:
    """
    Load model with correct architecture from hyperparameters or hardcoded defaults.

    Args:
        model_type: 'tcn', 'lstm', or 'cnn'
        weights_path: Path to model.pt or checkpoint
        hyperparams_path: Path to hyperparameters.json (None for hardcoded)
        device: Target device

    Returns:
        Loaded model with correct architecture
    """
    # Load hyperparameters
    if hyperparams_path is not None and hyperparams_path.exists():
        with open(hyperparams_path) as f:
            hyperparams = json.load(f)
        source = f"from {hyperparams_path.name}"
    else:
        hyperparams = get_hardcoded_hyperparams(model_type)
        source = "hardcoded (ablation)"

    # Create model with correct architecture
    model = create_model_from_hyperparams(model_type, hyperparams, device)

    # Load weights
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model


def soft_vote(probs_list: List[np.ndarray]) -> np.ndarray:
    """
    Simple soft voting: average probabilities across models.

    Args:
        probs_list: List of [N, num_classes] probability arrays

    Returns:
        [N, num_classes] averaged probabilities
    """
    return np.mean(probs_list, axis=0)


def weighted_vote(probs_list: List[np.ndarray], weights: List[float]) -> np.ndarray:
    """
    Weighted soft voting: weighted average of probabilities.

    Args:
        probs_list: List of [N, num_classes] probability arrays
        weights: List of weights (should sum to 1)

    Returns:
        [N, num_classes] weighted average probabilities
    """
    weights = np.array(weights) / np.sum(weights)  # Normalize
    result = np.zeros_like(probs_list[0])
    for probs, weight in zip(probs_list, weights):
        result += probs * weight
    return result


def hard_vote(preds_list: List[np.ndarray]) -> np.ndarray:
    """
    Hard voting: majority vote on predictions.

    Args:
        preds_list: List of [N] prediction arrays

    Returns:
        [N] majority vote predictions
    """
    stacked = np.stack(preds_list, axis=1)  # [N, num_models]
    # Mode (most common prediction) for each sample
    from scipy import stats
    result, _ = stats.mode(stacked, axis=1, keepdims=False)
    return result.flatten()


class ThermalEnsemble(nn.Module):
    """
    Ensemble of thermal classifiers using soft voting.

    Combines TCN, BiLSTM, and 1D CNN models for improved classification.
    """

    def __init__(
        self,
        tcn_model: nn.Module = None,
        lstm_model: nn.Module = None,
        cnn_model: nn.Module = None,
        weights: Optional[List[float]] = None
    ):
        """
        Initialize ensemble with pre-trained models.

        Args:
            tcn_model: Pre-trained TCN model
            lstm_model: Pre-trained BiLSTM model
            cnn_model: Pre-trained 1D CNN model
            weights: Optional weights for weighted voting (default: equal)
        """
        super().__init__()

        self.models = nn.ModuleDict()
        model_names = []

        if tcn_model is not None:
            self.models['tcn'] = tcn_model
            model_names.append('tcn')
        if lstm_model is not None:
            self.models['lstm'] = lstm_model
            model_names.append('lstm')
        if cnn_model is not None:
            self.models['cnn'] = cnn_model
            model_names.append('cnn')

        self.model_names = model_names
        self.num_models = len(model_names)

        if self.num_models == 0:
            raise ValueError("At least one model must be provided")

        # Weights for soft voting
        if weights is None:
            self.weights = [1.0 / self.num_models] * self.num_models
        else:
            if len(weights) != self.num_models:
                raise ValueError(f"Expected {self.num_models} weights, got {len(weights)}")
            self.weights = weights

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with soft voting.

        Args:
            x: Input tensor [batch, seq_len, 1]
            lengths: Sequence lengths [batch]
            mask: Optional mask [batch, seq_len]

        Returns:
            Ensemble logits [batch, num_classes]
        """
        all_logits = []

        for name, model in self.models.items():
            model.eval()
            with torch.no_grad():
                logits = model(x, lengths, mask)
            all_logits.append(logits)

        # Convert to probabilities, average, convert back to logits
        all_probs = [F.softmax(logits, dim=1) for logits in all_logits]

        # Weighted average
        ensemble_probs = torch.zeros_like(all_probs[0])
        for probs, weight in zip(all_probs, self.weights):
            ensemble_probs += probs * weight

        # Convert back to logits for compatibility with loss functions
        ensemble_logits = torch.log(ensemble_probs + 1e-10)

        return ensemble_logits

    def predict_proba(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Get ensemble probabilities and individual model probabilities.

        Args:
            x: Input tensor [batch, seq_len, 1]
            lengths: Sequence lengths [batch]
            mask: Optional mask [batch, seq_len]

        Returns:
            Tuple of (ensemble_probs, individual_probs_dict)
        """
        individual_probs = {}

        for name, model in self.models.items():
            model.eval()
            with torch.no_grad():
                logits = model(x, lengths, mask)
                probs = F.softmax(logits, dim=1)
            individual_probs[name] = probs

        # Weighted ensemble
        ensemble_probs = torch.zeros_like(list(individual_probs.values())[0])
        for (name, probs), weight in zip(individual_probs.items(), self.weights):
            ensemble_probs += probs * weight

        return ensemble_probs, individual_probs

    def get_num_parameters(self) -> int:
        """Get total number of parameters across all models."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def from_pretrained(
        cls,
        model_paths: Optional[Dict[str, Dict[str, Path]]] = None,
        device: torch.device = None,
        weights: Optional[List[float]] = None
    ) -> 'ThermalEnsemble':
        """
        Load ensemble from pretrained model checkpoints with dynamic architecture.

        Uses resolve_model_paths() to find models, preferring standard training
        outputs over ablation studies for fair comparison with RGB classifiers.

        Architecture is loaded from hyperparameters.json when available, falling
        back to hardcoded ablation configurations for backward compatibility.

        Args:
            model_paths: Optional override for model paths. If None, uses
                         resolve_model_paths() with standard > ablation priority.
                         Format: {'tcn': {'weights': Path, 'hyperparams': Path}, ...}
            device: Device to load models on
            weights: Optional weights for voting

        Returns:
            ThermalEnsemble instance
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Resolve paths (standard training first, then ablation fallback)
        if model_paths is None:
            model_paths = resolve_model_paths()

        # Load each model with dynamic architecture
        tcn_model = load_model_with_architecture(
            model_type='tcn',
            weights_path=model_paths['tcn']['weights'],
            hyperparams_path=model_paths['tcn'].get('hyperparams'),
            device=device
        )

        lstm_model = load_model_with_architecture(
            model_type='lstm',
            weights_path=model_paths['lstm']['weights'],
            hyperparams_path=model_paths['lstm'].get('hyperparams'),
            device=device
        )

        cnn_model = load_model_with_architecture(
            model_type='cnn',
            weights_path=model_paths['cnn']['weights'],
            hyperparams_path=model_paths['cnn'].get('hyperparams'),
            device=device
        )

        return cls(
            tcn_model=tcn_model,
            lstm_model=lstm_model,
            cnn_model=cnn_model,
            weights=weights
        )


def get_default_model_paths() -> Dict[str, Dict[str, Path]]:
    """
    Get default paths to best pretrained models with hyperparameters.

    This is a convenience wrapper around resolve_model_paths() that handles
    the FileNotFoundError gracefully for diagnostic purposes.

    Returns:
        Dict mapping model name to {'weights': Path, 'hyperparams': Path or None}

    Note:
        For actual model loading, use resolve_model_paths() directly which
        raises FileNotFoundError with helpful diagnostics.
    """
    return resolve_model_paths()


def load_default_ensemble(
    device: torch.device = None,
    weights: Optional[List[float]] = None
) -> ThermalEnsemble:
    """
    Load ensemble with default best models using dynamic architecture loading.

    Prefers standard training outputs (tcn/, lstm/, cnn_1d/) over ablation
    studies for fair comparison with RGB classifiers. Architecture is loaded
    from hyperparameters.json when available.

    Args:
        device: Device to load models on
        weights: Optional weights (default: equal weighting)

    Returns:
        ThermalEnsemble instance

    Raises:
        FileNotFoundError: If required model files not found
    """
    # resolve_model_paths() raises FileNotFoundError with helpful diagnostics
    model_paths = resolve_model_paths()

    # Default to equal weights (fair comparison, no prior F1 bias)
    if weights is None:
        weights = [1/3, 1/3, 1/3]

    return ThermalEnsemble.from_pretrained(
        model_paths=model_paths,
        device=device,
        weights=weights
    )


if __name__ == '__main__':
    # Test ensemble loading with dynamic architecture
    print("Testing ThermalEnsemble with dynamic architecture loading...")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Check model paths with new resolve_model_paths()
    print("\nResolving model paths (standard training preferred)...")
    try:
        paths = resolve_model_paths()
        print("\nModel paths resolved:")
        for name, path_info in paths.items():
            weights_path = path_info['weights']
            hyperparams_path = path_info.get('hyperparams')
            exists_w = "✓" if weights_path.exists() else "✗"
            print(f"  {name}:")
            print(f"    weights: {exists_w} {weights_path}")
            if hyperparams_path:
                exists_h = "✓" if hyperparams_path.exists() else "✗"
                print(f"    hyperparams: {exists_h} {hyperparams_path}")
            else:
                print(f"    hyperparams: (hardcoded ablation config)")
    except FileNotFoundError as e:
        print(f"\n✗ Could not resolve paths: {e}")
        print("\nRun the following to generate model files:")
        print("  python -m classifiers.thermal_classifiers.run_all --methods tcn lstm cnn_1d --n-trials 2")
        sys.exit(1)

    # Load ensemble with dynamic architecture
    try:
        print("\nLoading ensemble with dynamic architecture...")
        ensemble = load_default_ensemble(device=device)
        print(f"\n✓ Ensemble loaded successfully!")
        print(f"  Models: {ensemble.model_names}")
        print(f"  Weights: {[f'{w:.4f}' for w in ensemble.weights]}")
        print(f"  Total parameters: {ensemble.get_num_parameters():,}")

        # Test forward pass
        batch_size = 4
        seq_len = 1000
        x = torch.randn(batch_size, seq_len, 1).to(device)
        lengths = torch.full((batch_size,), seq_len).to(device)

        logits = ensemble(x, lengths)
        print(f"\nForward pass test:")
        print(f"  Input: {x.shape}")
        print(f"  Output: {logits.shape}")
        print(f"  Predictions: {logits.argmax(dim=1).tolist()}")
        print(f"  Class names: {[CLASS_NAMES[i] for i in logits.argmax(dim=1).tolist()]}")

    except FileNotFoundError as e:
        print(f"\n✗ Could not load ensemble: {e}")
        print("\nRun tcn, lstm, cnn_1d methods first to generate model files:")
        print("  python -m classifiers.thermal_classifiers.run_all --methods tcn lstm cnn_1d --n-trials 2")
    except Exception as e:
        print(f"\n✗ Error loading ensemble: {e}")
        import traceback
        traceback.print_exc()
