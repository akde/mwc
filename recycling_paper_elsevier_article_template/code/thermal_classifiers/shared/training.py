"""
Training utilities for thermal time series classification.

This module provides:
- EarlyStopping: Callback for early stopping with patience
- train_epoch: Single epoch training function
- evaluate: Model evaluation function
- train_model: Full training loop with validation and checkpointing
- save/load checkpoint utilities
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, Callable
import json
from datetime import datetime
import copy

from .config import (
    RANDOM_SEED, MAX_EPOCHS, EARLY_STOPPING_PATIENCE, EARLY_STOPPING_MIN_DELTA,
    WEIGHT_DECAY, GRADIENT_CLIP_NORM, LOG_INTERVAL, SAVE_INTERVAL, set_seed
)


class EarlyStopping:
    """
    Early stopping callback to stop training when validation loss stops improving.

    Monitors a metric and stops training if no improvement is seen for `patience` epochs.
    """

    def __init__(
        self,
        patience: int = EARLY_STOPPING_PATIENCE,
        min_delta: float = EARLY_STOPPING_MIN_DELTA,
        mode: str = 'min',
        verbose: bool = True
    ):
        """
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy/F1
            verbose: Print early stopping messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.best_score = None
        self.counter = 0
        self.should_stop = False
        self.best_epoch = 0

    def __call__(self, score: float, epoch: int) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current metric value
            epoch: Current epoch number

        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False

        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: No improvement for {self.counter}/{self.patience} epochs")

        if self.counter >= self.patience:
            self.should_stop = True
            if self.verbose:
                print(f"Early stopping triggered at epoch {epoch}")
                print(f"Best score: {self.best_score:.4f} at epoch {self.best_epoch}")

        return self.should_stop


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    gradient_clip: float = GRADIENT_CLIP_NORM,
    log_interval: int = LOG_INTERVAL
) -> Dict[str, float]:
    """
    Train model for one epoch.

    Args:
        model: PyTorch model
        train_loader: Training DataLoader
        optimizer: Optimizer
        criterion: Loss function
        device: Torch device
        gradient_clip: Max gradient norm
        log_interval: Print progress every N batches

    Returns:
        Dict with 'loss' and 'accuracy'
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (sequences, labels, lengths, masks, _) in enumerate(train_loader):
        sequences = sequences.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        # Forward pass - models expect (sequences, lengths, masks) for beginning padding
        outputs = model(sequences, lengths, masks)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()

        # Gradient clipping
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        optimizer.step()

        # Track metrics
        total_loss += loss.item() * len(labels)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += len(labels)

        if log_interval > 0 and (batch_idx + 1) % log_interval == 0:
            print(f"  Batch {batch_idx + 1}/{len(train_loader)}: "
                  f"Loss={loss.item():.4f}, Acc={100.*correct/total:.2f}%")

    return {
        'loss': total_loss / total,
        'accuracy': correct / total
    }


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    return_predictions: bool = False,
    compute_f1: bool = True
) -> Dict:
    """
    Evaluate model on a dataset.

    Args:
        model: PyTorch model
        data_loader: DataLoader
        criterion: Loss function
        device: Torch device
        return_predictions: Also return predictions and labels
        compute_f1: Compute F1-weighted score (required for fair early stopping)

    Returns:
        Dict with 'loss', 'accuracy', 'f1_weighted', and optionally 'predictions', 'labels', 'unique_ids', 'probabilities'
    """
    from sklearn.metrics import f1_score

    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    all_predictions = []
    all_labels = []
    all_unique_ids = []
    all_probabilities = []

    with torch.no_grad():
        for sequences, labels, lengths, masks, unique_ids in data_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)
            masks = masks.to(device)

            outputs = model(sequences, lengths, masks)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * len(labels)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += len(labels)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            if return_predictions:
                all_unique_ids.extend(unique_ids)
                # Compute softmax probabilities for RGB-aligned output
                probs = torch.softmax(outputs, dim=1)
                all_probabilities.extend(probs.cpu().numpy())

    result = {
        'loss': total_loss / total,
        'accuracy': correct / total
    }

    # Compute F1-weighted for early stopping (ALIGNED WITH RGB for fair comparison)
    if compute_f1:
        result['f1_weighted'] = f1_score(
            all_labels, all_predictions, average='weighted', zero_division=0
        )

    if return_predictions:
        result['predictions'] = np.array(all_predictions)
        result['labels'] = np.array(all_labels)
        result['unique_ids'] = all_unique_ids
        result['probabilities'] = np.array(all_probabilities)

    return result


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: torch.Tensor,
    device: torch.device,
    learning_rate: float = 1e-3,
    max_epochs: int = MAX_EPOCHS,
    patience: int = EARLY_STOPPING_PATIENCE,
    weight_decay: float = WEIGHT_DECAY,
    checkpoint_dir: Optional[Path] = None,
    model_name: str = 'model',
    verbose: bool = True,
    label_smoothing: float = 0.0,  # NEW: reduce overconfident predictions
) -> Dict:
    """
    Full training loop with validation and early stopping.

    Args:
        model: PyTorch model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        class_weights: Tensor of class weights for loss function
        device: Torch device
        learning_rate: Initial learning rate
        max_epochs: Maximum training epochs
        patience: Early stopping patience
        weight_decay: Weight decay for AdamW
        checkpoint_dir: Directory to save checkpoints
        model_name: Name for saved model files
        verbose: Print training progress
        label_smoothing: Label smoothing factor (0-0.2 recommended)

    Returns:
        Dict with training history and best model state
    """
    set_seed(RANDOM_SEED)

    # Setup
    model = model.to(device)
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # Use F1-weighted for scheduling and early stopping (ALIGNED WITH RGB for fair comparison)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5, verbose=verbose)
    early_stopping = EarlyStopping(patience=patience, mode='max', verbose=verbose)

    # Tracking
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [],
        'lr': []
    }
    best_val_f1 = 0.0
    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = 0

    if checkpoint_dir:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\nStarting training for {model_name}")
        print(f"  Device: {device}")
        print(f"  Max epochs: {max_epochs}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Early stopping patience: {patience}")
        print()

    for epoch in range(1, max_epochs + 1):
        if verbose:
            print(f"Epoch {epoch}/{max_epochs}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device,
            log_interval=LOG_INTERVAL if verbose else 0
        )

        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device)

        # Track metrics
        current_lr = optimizer.param_groups[0]['lr']
        val_f1 = val_metrics.get('f1_weighted', 0.0)

        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_f1)
        history['lr'].append(current_lr)

        if verbose:
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {100*train_metrics['accuracy']:.2f}%")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {100*val_metrics['accuracy']:.2f}%, F1: {val_f1:.4f}")
            print(f"  LR: {current_lr:.2e}")

        # Save best model based on F1-weighted (ALIGNED WITH RGB for fair comparison)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_loss = val_metrics['loss']
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch

            if checkpoint_dir:
                save_checkpoint(
                    model, optimizer, epoch, val_metrics['loss'],
                    checkpoint_dir / f'{model_name}_best.pt'
                )
                if verbose:
                    print(f"  Saved best model (val_f1: {best_val_f1:.4f})")

        # Periodic checkpoint
        if checkpoint_dir and epoch % SAVE_INTERVAL == 0:
            save_checkpoint(
                model, optimizer, epoch, val_metrics['loss'],
                checkpoint_dir / f'{model_name}_epoch{epoch}.pt'
            )

        # Learning rate scheduling based on F1 (mode='max')
        scheduler.step(val_f1)

        # Early stopping based on F1-weighted (ALIGNED WITH RGB for fair comparison)
        if early_stopping(val_f1, epoch):
            break

        if verbose:
            print()

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return {
        'history': history,
        'best_val_loss': best_val_loss,
        'best_val_f1': best_val_f1,
        'best_epoch': best_epoch,
        'final_epoch': epoch,
        'model_state': best_model_state
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    path: Path
):
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'timestamp': datetime.now().isoformat()
    }, path)


def load_checkpoint(
    model: nn.Module,
    path: Path,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device = None
) -> Dict:
    """
    Load model checkpoint.

    Args:
        model: PyTorch model to load weights into
        path: Path to checkpoint file
        optimizer: Optional optimizer to load state into
        device: Device to load to

    Returns:
        Dict with checkpoint info (epoch, val_loss, timestamp)
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return {
        'epoch': checkpoint['epoch'],
        'val_loss': checkpoint['val_loss'],
        'timestamp': checkpoint.get('timestamp', 'unknown')
    }


def save_training_results(
    results: Dict,
    model_name: str,
    output_dir: Path
):
    """Save training results to JSON file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert numpy types to Python types for JSON serialization
    serializable = {}
    for key, value in results.items():
        if key == 'model_state':
            continue  # Don't save model state to JSON
        if isinstance(value, dict):
            serializable[key] = {k: float(v) if isinstance(v, (np.floating, float)) else v
                                 for k, v in value.items()}
        elif isinstance(value, list):
            serializable[key] = [float(v) if isinstance(v, (np.floating, float)) else v
                                 for v in value]
        else:
            serializable[key] = float(value) if isinstance(value, (np.floating, float)) else value

    serializable['model_name'] = model_name
    serializable['timestamp'] = datetime.now().isoformat()

    with open(output_dir / f'{model_name}_training_results.json', 'w') as f:
        json.dump(serializable, f, indent=2)
