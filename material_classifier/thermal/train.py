"""Thermal feature caching and training entry point.

Caches DINOv2 [CLS] features from thermal frames, then trains
attention pool + MLP head on cached features.
"""

import argparse
import os

import torch
import yaml
from tqdm import tqdm

from material_classifier.data.preprocessing import get_transform
from material_classifier.thermal.dataset import ThermalTrackletDataset
from material_classifier.train import train


def cache_thermal_features(config):
    """
    Extract DINOv2 [CLS] features for all thermal tracklets.
    Saves to thermal_features/{video_stem}_track_{track_id}.pt -> (T, 1024).
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

    thermal_shape = tuple(data_cfg.get("thermal_shape", [164, 270]))
    colormap = data_cfg.get("colormap", None)
    downloads_dir = data_cfg.get("downloads_dir", "~/Downloads")

    # Process all splits
    for split in ["train", "val", "test"]:
        try:
            dataset = ThermalTrackletDataset(
                labels_csv=data_cfg["labels_csv"],
                tracklets_dir=data_cfg["tracklets_dir"],
                downloads_dir=downloads_dir,
                split=split,
                num_frames=data_cfg["num_frames"],
                transform=transform,
                thermal_shape=thermal_shape,
                colormap=colormap,
            )
        except Exception:
            continue

        if len(dataset) == 0:
            continue

        print(f"Caching thermal features for {split} split ({len(dataset)} tracklets)...")

        for idx in tqdm(range(len(dataset)), desc=f"  {split}"):
            video, track_id, _ = dataset.samples[idx]
            video_stem = os.path.splitext(video)[0]
            feature_path = os.path.join(
                features_dir,
                f"{video_stem}_track_{track_id}.pt",
            )

            # Skip if already cached
            if os.path.exists(feature_path):
                continue

            frames, _ = dataset[idx]  # (T, 3, 518, 518)
            frames = frames.to(device)

            with torch.amp.autocast("cuda"):
                features = extractor(frames)  # (T, 1024)

            torch.save(features.cpu(), feature_path)

    print("Thermal feature caching complete.")


def main():
    parser = argparse.ArgumentParser(description="Train thermal material classifier")
    parser.add_argument(
        "--config",
        default="material_classifier/config/thermal.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--cache-features",
        action="store_true",
        help="Run feature caching only (no training)",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.cache_features:
        cache_thermal_features(config)
    else:
        cache_thermal_features(config)
        train(config)


if __name__ == "__main__":
    main()
