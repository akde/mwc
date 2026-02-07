import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix,
)
from torch.utils.data import DataLoader

from material_classifier.data.dataset import (
    CachedFeatureDataset, collate_fn, CLASS_NAMES,
)
from material_classifier.models.pipeline import CachedMaterialClassifier


def evaluate(model, dataloader, device, class_names):
    """
    Full evaluation on a dataset split.

    Returns:
        dict with accuracy, macro_f1, per_class metrics, confusion_matrix,
        all_preds, all_labels.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, masks, labels in dataloader:
            features = features.to(device)
            masks = masks.to(device)

            logits = model(features, masks)
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, labels=list(range(len(class_names))),
        zero_division=0,
    )
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))

    per_class = {}
    for i, name in enumerate(class_names):
        per_class[name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
        }

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": cm,
        "all_preds": all_preds,
        "all_labels": all_labels,
    }


def plot_confusion_matrix(cm, class_names, save_path):
    """Save confusion matrix as PNG."""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate material classifier")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--split", default="test", help="Dataset split to evaluate")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_cfg = config["data"]
    model_cfg = config["model"]
    output_cfg = config["output"]

    # Load dataset
    dataset = CachedFeatureDataset(
        labels_csv=data_cfg["labels_csv"],
        features_dir=data_cfg["features_dir"],
        split=args.split,
    )
    dataloader = DataLoader(
        dataset, batch_size=config["training"]["batch_size"],
        shuffle=False, num_workers=4, collate_fn=collate_fn,
        pin_memory=True,
    )

    # Determine feat_dim from checkpoint or sample
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Infer feat_dim from checkpoint
    state_dict = checkpoint["model_state_dict"]
    # pool.attn_proj.0.weight has shape (dim//4, dim)
    feat_dim = state_dict["pool.attn_proj.0.weight"].shape[1]

    model = CachedMaterialClassifier(
        feat_dim=feat_dim,
        hidden_dim=model_cfg["hidden_dim"],
        num_classes=model_cfg["num_classes"],
        dropout=model_cfg["dropout"],
        temperature=model_cfg.get("attention_temperature", 1.0),
    ).to(device)
    model.load_state_dict(state_dict)

    # Evaluate
    results = evaluate(model, dataloader, device, CLASS_NAMES)

    # Print results
    print(f"\n{'='*50}")
    print(f"Evaluation on {args.split} split ({len(dataset)} samples)")
    print(f"{'='*50}")
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Macro F1:  {results['macro_f1']:.4f}")
    print(f"\nPer-class metrics:")
    for name in CLASS_NAMES:
        m = results["per_class"][name]
        print(f"  {name:8s}  P={m['precision']:.4f}  R={m['recall']:.4f}  F1={m['f1']:.4f}")

    print(f"\nConfusion Matrix:")
    print(results["confusion_matrix"])

    # Save confusion matrix plot
    os.makedirs(output_cfg["log_dir"], exist_ok=True)
    cm_path = os.path.join(output_cfg["log_dir"], f"confusion_matrix_{args.split}.png")
    plot_confusion_matrix(results["confusion_matrix"], CLASS_NAMES, cm_path)


if __name__ == "__main__":
    main()
