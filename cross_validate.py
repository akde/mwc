"""5-fold stratified cross-validation for material classifier."""

import csv
import os
import copy

import numpy as np
import torch
import yaml
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torch.utils.data import DataLoader

from material_classifier.data.dataset import CachedFeatureDataset, collate_fn, CLASS_NAMES
from material_classifier.models.pipeline import CachedMaterialClassifier
from material_classifier.train import set_seed


def load_labels(labels_csv):
    with open(labels_csv) as f:
        return list(csv.DictReader(f))


def write_labels(labels_csv, rows):
    with open(labels_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video", "track_id", "split", "class"])
        writer.writeheader()
        writer.writerows(rows)


def train_one_fold(config, fold_idx):
    """Train and return best model state dict + best val f1."""
    seed = config["training"]["seed"] + fold_idx
    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]

    train_dataset = CachedFeatureDataset(
        labels_csv=data_cfg["labels_csv"],
        features_dir=data_cfg["features_dir"],
        split="train",
    )
    val_dataset = CachedFeatureDataset(
        labels_csv=data_cfg["labels_csv"],
        features_dir=data_cfg["features_dir"],
        split="val",
    )

    train_loader = DataLoader(
        train_dataset, batch_size=train_cfg["batch_size"],
        shuffle=True, num_workers=4, collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=train_cfg["batch_size"],
        shuffle=False, num_workers=4, collate_fn=collate_fn,
        pin_memory=True,
    )

    sample_features, _ = train_dataset[0]
    feat_dim = sample_features.shape[-1]

    model = CachedMaterialClassifier(
        feat_dim=feat_dim,
        hidden_dim=model_cfg["hidden_dim"],
        num_classes=model_cfg["num_classes"],
        dropout=model_cfg["dropout"],
        temperature=model_cfg.get("attention_temperature", 1.0),
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=train_cfg["label_smoothing"])

    warmup_epochs = train_cfg["warmup_epochs"]
    total_epochs = train_cfg["epochs"]
    warmup_scheduler = LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs)
    scheduler = SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

    best_f1 = 0.0
    best_state = None

    for epoch in range(total_epochs):
        model.train()
        for features, masks, labels in train_loader:
            features, masks, labels = features.to(device), masks.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(features, masks)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        # Validate
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for features, masks, labels in val_loader:
                features, masks = features.to(device), masks.to(device)
                preds = model(features, masks).argmax(dim=-1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.tolist())

        val_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())

        scheduler.step()

    return best_state, best_f1


def evaluate_fold(config, state_dict):
    """Evaluate a trained model on the val split, return predictions and labels."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_cfg = config["data"]
    model_cfg = config["model"]

    val_dataset = CachedFeatureDataset(
        labels_csv=data_cfg["labels_csv"],
        features_dir=data_cfg["features_dir"],
        split="val",
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["training"]["batch_size"],
        shuffle=False, num_workers=4, collate_fn=collate_fn,
        pin_memory=True,
    )

    sample_features, _ = val_dataset[0]
    feat_dim = sample_features.shape[-1]

    model = CachedMaterialClassifier(
        feat_dim=feat_dim,
        hidden_dim=model_cfg["hidden_dim"],
        num_classes=model_cfg["num_classes"],
        dropout=model_cfg["dropout"],
        temperature=model_cfg.get("attention_temperature", 1.0),
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for features, masks, labels in val_loader:
            features, masks = features.to(device), masks.to(device)
            preds = model(features, masks).argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())

    return all_preds, all_labels


def main():
    config_path = "material_classifier/config/default.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    labels_csv = config["data"]["labels_csv"]
    rows = load_labels(labels_csv)

    # Extract classes for stratification
    classes = [r["class"] for r in rows]
    class_to_idx = {c: i for i, c in enumerate(CLASS_NAMES)}
    y = np.array([class_to_idx[c] for c in classes])

    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    all_preds = []
    all_labels = []
    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(rows, y)):
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx + 1}/{n_folds}  (train={len(train_idx)}, val={len(val_idx)})")
        print(f"{'='*60}")

        # Write fold splits to labels.csv
        fold_rows = []
        for i, r in enumerate(rows):
            row = dict(r)
            row["split"] = "train" if i in set(train_idx) else "val"
            fold_rows.append(row)
        write_labels(labels_csv, fold_rows)

        # Train
        best_state, best_f1 = train_one_fold(config, fold_idx)
        print(f"  Best val F1 during training: {best_f1:.4f}")

        # Evaluate
        preds, labels = evaluate_fold(config, best_state)
        all_preds.extend(preds)
        all_labels.extend(labels)

        # Per-fold metrics
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="macro", zero_division=0)
        fold_results.append({"fold": fold_idx + 1, "accuracy": acc, "macro_f1": f1, "n": len(labels)})
        print(f"  Fold {fold_idx + 1} — Accuracy: {acc:.4f}, Macro F1: {f1:.4f}")

    # Aggregate results
    print(f"\n{'='*60}")
    print(f"5-Fold Cross-Validation Results ({len(all_labels)} total samples)")
    print(f"{'='*60}")

    for fr in fold_results:
        print(f"  Fold {fr['fold']}: Acc={fr['accuracy']:.4f}  F1={fr['macro_f1']:.4f}  (n={fr['n']})")

    accs = [fr["accuracy"] for fr in fold_results]
    f1s = [fr["macro_f1"] for fr in fold_results]
    print(f"\n  Mean Accuracy: {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
    print(f"  Mean Macro F1: {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")

    # Overall (all predictions pooled)
    overall_acc = accuracy_score(all_labels, all_preds)
    overall_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    precision, recall, f1_per, _ = precision_recall_fscore_support(
        all_labels, all_preds, labels=list(range(len(CLASS_NAMES))), zero_division=0,
    )
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(CLASS_NAMES))))

    print(f"\n  Pooled Accuracy: {overall_acc:.4f}")
    print(f"  Pooled Macro F1: {overall_f1:.4f}")
    print(f"\n  Per-class (pooled across all folds):")
    for i, name in enumerate(CLASS_NAMES):
        print(f"    {name:8s}  P={precision[i]:.4f}  R={recall[i]:.4f}  F1={f1_per[i]:.4f}")

    print(f"\n  Confusion Matrix (pooled):")
    print(cm)

    # Restore original 80/20 split
    from material_classifier.label import assign_splits
    assign_splits(labels_csv)
    print(f"\nRestored original 80/20 train/val split in {labels_csv}")


if __name__ == "__main__":
    main()
