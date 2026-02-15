import csv
import os

import torch
from torch.utils.data import Dataset

from material_classifier.data.dataset import CLASS_TO_IDX


class FusedCachedFeatureDataset(Dataset):
    """
    Dataset that loads paired RGB + thermal precomputed DINOv2 features.
    Both modalities must have matching feature files for every tracklet.
    """

    def __init__(self, labels_csv, rgb_features_dir, thermal_features_dir, split):
        self.rgb_features_dir = rgb_features_dir
        self.thermal_features_dir = thermal_features_dir
        self.samples = []  # (rgb_path, thermal_path, label)

        with open(labels_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] == split:
                    video_stem = os.path.splitext(row["video"])[0]
                    fname = f"{video_stem}_track_{row['track_id']}.pt"
                    rgb_path = os.path.join(rgb_features_dir, fname)
                    thermal_path = os.path.join(thermal_features_dir, fname)
                    self.samples.append((
                        rgb_path,
                        thermal_path,
                        CLASS_TO_IDX[row["class"]],
                    ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rgb_path, thermal_path, label = self.samples[idx]
        rgb_features = torch.load(rgb_path, weights_only=True)        # (T, 1024)
        thermal_features = torch.load(thermal_path, weights_only=True)  # (T, 1024)
        return rgb_features, thermal_features, label


def fused_collate_fn(batch):
    """
    Custom collate for fused RGB + thermal variable-length sequences.
    Pads both modalities independently to their max length in the batch.

    Returns:
        rgb_padded: (B, max_T_rgb, D)
        rgb_masks: (B, max_T_rgb) boolean, True = valid frame
        thermal_padded: (B, max_T_thermal, D)
        thermal_masks: (B, max_T_thermal) boolean, True = valid frame
        labels: (B,) long tensor
    """
    B = len(batch)
    max_T_rgb = max(rgb.shape[0] for rgb, _, _ in batch)
    max_T_thermal = max(th.shape[0] for _, th, _ in batch)
    feat_dim = batch[0][0].shape[-1]

    rgb_padded = torch.zeros(B, max_T_rgb, feat_dim)
    rgb_masks = torch.zeros(B, max_T_rgb, dtype=torch.bool)
    thermal_padded = torch.zeros(B, max_T_thermal, feat_dim)
    thermal_masks = torch.zeros(B, max_T_thermal, dtype=torch.bool)
    labels = torch.zeros(B, dtype=torch.long)

    for i, (rgb, thermal, label) in enumerate(batch):
        T_rgb = rgb.shape[0]
        rgb_padded[i, :T_rgb] = rgb
        rgb_masks[i, :T_rgb] = True

        T_thermal = thermal.shape[0]
        thermal_padded[i, :T_thermal] = thermal
        thermal_masks[i, :T_thermal] = True

        labels[i] = label

    return rgb_padded, rgb_masks, thermal_padded, thermal_masks, labels
