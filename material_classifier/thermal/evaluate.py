"""Thermal model evaluation entry point."""

import argparse
import os

import torch
import yaml
from torch.utils.data import DataLoader

from material_classifier.data.dataset import CachedFeatureDataset, collate_fn, CLASS_NAMES
from material_classifier.evaluate import evaluate, plot_confusion_matrix
from material_classifier.models.pipeline import CachedMaterialClassifier


def main():
    parser = argparse.ArgumentParser(description="Evaluate thermal material classifier")
    parser.add_argument(
        "--config",
        default="material_classifier/config/thermal.yaml",
        help="Path to config YAML",
    )
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--split", default="val", help="Dataset split to evaluate")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_cfg = config["data"]
    model_cfg = config["model"]
    output_cfg = config["output"]

    # Load dataset (cached thermal features)
    dataset = CachedFeatureDataset(
        labels_csv=data_cfg["labels_csv"],
        features_dir=data_cfg["features_dir"],
        split=args.split,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Load checkpoint and infer feat_dim
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"]
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
    print(f"Thermal evaluation on {args.split} split ({len(dataset)} samples)")
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
