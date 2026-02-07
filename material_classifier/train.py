import argparse
import os
import random

import numpy as np
import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from material_classifier.data.dataset import (
    TrackletDataset, CachedFeatureDataset, collate_fn, CLASS_NAMES,
)
from material_classifier.data.preprocessing import get_transform
from material_classifier.models.pipeline import CachedMaterialClassifier


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cache_features(config):
    """
    One-time preprocessing: extract DINOv2 [CLS] features for all tracklets.
    Saves to features/{video_stem}_track_{track_id}.pt -> (T, 1024).
    """
    from material_classifier.models.feature_extractor import DINOv2Extractor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_cfg = config["data"]
    model_cfg = config["model"]

    features_dir = data_cfg["features_dir"]
    os.makedirs(features_dir, exist_ok=True)

    # Load DINOv2
    extractor = DINOv2Extractor(model_cfg["backbone"]).to(device)

    # Build dataset (no augmentation for feature caching)
    transform = get_transform(image_size=data_cfg["image_size"], train=False)

    # Process all splits
    for split in ["train", "val", "test"]:
        try:
            dataset = TrackletDataset(
                labels_csv=data_cfg["labels_csv"],
                tracklets_dir=data_cfg["tracklets_dir"],
                videos_dir=data_cfg["videos_dir"],
                split=split,
                num_frames=data_cfg["num_frames"],
                transform=transform,
            )
        except Exception:
            continue

        if len(dataset) == 0:
            continue

        print(f"Caching features for {split} split ({len(dataset)} tracklets)...")

        for idx in tqdm(range(len(dataset)), desc=f"  {split}"):
            video, track_id, _ = dataset.samples[idx]
            video_stem = os.path.splitext(video)[0]
            feature_path = os.path.join(
                features_dir,
                f"{video_stem}_track_{track_id}.pt"
            )

            # Skip if already cached
            if os.path.exists(feature_path):
                continue

            frames, _ = dataset[idx]  # (T, 3, 518, 518)
            frames = frames.to(device)

            with torch.amp.autocast("cuda"):
                features = extractor(frames)  # (T, 1024)

            torch.save(features.cpu(), feature_path)

    print("Feature caching complete.")


def train(config):
    """Training loop for attention pool + MLP head on cached features."""
    seed = config["training"]["seed"]
    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]
    output_cfg = config["output"]

    os.makedirs(output_cfg["checkpoint_dir"], exist_ok=True)
    os.makedirs(output_cfg["log_dir"], exist_ok=True)

    # Datasets
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
        pin_memory=True, drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=train_cfg["batch_size"],
        shuffle=False, num_workers=4, collate_fn=collate_fn,
        pin_memory=True, drop_last=False,
    )

    # Model
    # Determine feature dim from a sample
    sample_features, _ = train_dataset[0]
    feat_dim = sample_features.shape[-1]

    model = CachedMaterialClassifier(
        feat_dim=feat_dim,
        hidden_dim=model_cfg["hidden_dim"],
        num_classes=model_cfg["num_classes"],
        dropout=model_cfg["dropout"],
        temperature=model_cfg.get("attention_temperature", 1.0),
    ).to(device)

    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Optimizer, loss, scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=train_cfg["label_smoothing"])

    warmup_epochs = train_cfg["warmup_epochs"]
    total_epochs = train_cfg["epochs"]
    warmup_scheduler = LinearLR(
        optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_epochs,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=total_epochs - warmup_epochs,
    )
    scheduler = SequentialLR(
        optimizer, [warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs],
    )

    grad_accum = train_cfg.get("grad_accumulation_steps", 1)
    best_f1 = 0.0

    for epoch in range(total_epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs} [train]")
        for step, (features, masks, labels) in enumerate(pbar):
            features = features.to(device)
            masks = masks.to(device)
            labels = labels.to(device)

            logits = model(features, masks)
            loss = criterion(logits, labels)
            loss = loss / grad_accum
            loss.backward()

            if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item() * grad_accum
            preds = logits.argmax(dim=-1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

            pbar.set_postfix(loss=f"{loss.item() * grad_accum:.4f}")

        train_acc = train_correct / max(train_total, 1)

        # Validate
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for features, masks, labels in val_loader:
                features = features.to(device)
                masks = masks.to(device)
                labels = labels.to(device)

                logits = model(features, masks)
                loss = criterion(logits, labels)
                val_loss += loss.item()

                preds = logits.argmax(dim=-1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        val_loss /= max(len(val_loader), 1)

        # Compute metrics
        from sklearn.metrics import accuracy_score, f1_score
        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1}/{total_epochs} | "
              f"train_loss={train_loss/max(len(train_loader),1):.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f} | "
              f"lr={lr:.6f}")

        # Save best
        if val_f1 > best_f1:
            best_f1 = val_f1
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_f1": best_f1,
                "config": config,
            }
            path = os.path.join(output_cfg["checkpoint_dir"], "best_model.pt")
            torch.save(checkpoint, path)
            print(f"  -> Saved best model (f1={best_f1:.4f})")

        # Save last
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_f1": best_f1,
            "config": config,
        }
        torch.save(
            checkpoint,
            os.path.join(output_cfg["checkpoint_dir"], "last_model.pt"),
        )

        scheduler.step()

    print(f"\nTraining complete. Best val F1: {best_f1:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train material classifier")
    parser.add_argument("--config", default="material_classifier/config/default.yaml",
                        help="Path to config YAML")
    parser.add_argument("--cache-features", action="store_true",
                        help="Run feature caching only (no training)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.cache_features:
        cache_features(config)
    else:
        cache_features(config)
        train(config)


if __name__ == "__main__":
    main()
