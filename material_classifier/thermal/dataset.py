"""Thermal tracklet dataset for DINOv2 feature extraction."""

import csv
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from material_classifier.data.dataset import CLASS_TO_IDX
from material_classifier.data.preprocessing import apply_mask_and_crop, get_transform, sample_frames
from material_classifier.thermal.utils import (
    get_experiment_id,
    grayscale_to_3channel,
    load_experiment_resources,
    warp_mask_to_thermal,
)


class ThermalTrackletDataset(Dataset):
    """
    Dataset that loads thermal frames + warped masks for each tracklet.

    For each tracklet frame:
    1. Look up the corresponding thermal frame via RGB-thermal frame matching
    2. Load the thermal frame (grayscale)
    3. Warp the RGB mask to thermal space via homography
    4. Convert grayscale to 3-channel (replicate or colormap)
    5. Apply mask+crop with gray fill
    6. Resize to 518x518 and normalize (ImageNet)
    """

    def __init__(
        self,
        labels_csv,
        tracklets_dir,
        downloads_dir,
        split,
        num_frames=8,
        transform=None,
        aug_config=None,
        image_size=518,
        thermal_shape=(164, 270),
        colormap=None,
    ):
        self.tracklets_dir = tracklets_dir
        self.downloads_dir = os.path.expanduser(downloads_dir)
        self.num_frames = num_frames
        self.transform = transform
        self.aug_config = aug_config
        self.image_size = image_size
        self.thermal_shape = tuple(thermal_shape)
        self.colormap = colormap

        # Parse labels.csv and filter by split
        self.samples = []  # (video, track_id, label)
        with open(labels_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] == split:
                    self.samples.append((
                        row["video"],
                        int(row["track_id"]),
                        CLASS_TO_IDX[row["class"]],
                    ))

        # Cache per-experiment resources: {exp_id: {H, frame_map, thermal_frames_dir}}
        self._exp_resources = {}
        exp_ids_needed = set()
        for video, _, _ in self.samples:
            exp_ids_needed.add(get_experiment_id(video))

        missing = []
        for exp_id in sorted(exp_ids_needed):
            exp_dir = os.path.join(self.downloads_dir, exp_id)
            res = load_experiment_resources(exp_dir, exp_id)
            if res is None:
                missing.append(exp_id)
            else:
                self._exp_resources[exp_id] = res

        if missing:
            print(f"  Warning: thermal resources missing for experiments: {missing}")

        # Build per-tracklet matched frame index
        # {(video, track_id): [(thermal_path, mask_path, rgb_frame_idx), ...]}
        self.tracklet_frames = {}
        for video, track_id, _ in self.samples:
            key = (video, track_id)
            if key in self.tracklet_frames:
                continue

            exp_id = get_experiment_id(video)
            res = self._exp_resources.get(exp_id)
            if res is None:
                self.tracklet_frames[key] = []
                continue

            video_stem = os.path.splitext(video)[0]
            csv_path = os.path.join(tracklets_dir, video_stem, "tracklet_data.csv")
            masks_dir = os.path.join(tracklets_dir, video_stem, "masks")

            if not os.path.exists(csv_path):
                self.tracklet_frames[key] = []
                continue

            frame_map = res["frame_map"]
            thermal_dir = res["thermal_frames_dir"]

            frames = []
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if int(row["track_id"]) != track_id:
                        continue

                    rgb_frame_idx = int(row["frame"])

                    # Look up thermal frame
                    thermal_frame_idx = frame_map.get(rgb_frame_idx)
                    if thermal_frame_idx is None:
                        continue

                    # Check thermal file exists
                    thermal_path = os.path.join(
                        thermal_dir, f"FLIR_frame_{thermal_frame_idx:06d}.png"
                    )
                    if not os.path.exists(thermal_path):
                        continue

                    # Check mask file exists
                    mask_path = os.path.join(
                        masks_dir, f"frame_{rgb_frame_idx:06d}_track_{track_id}.png"
                    )
                    if not os.path.exists(mask_path):
                        continue

                    frames.append((thermal_path, mask_path, rgb_frame_idx))

            self.tracklet_frames[key] = frames

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video, track_id, label = self.samples[idx]
        key = (video, track_id)
        all_frames = self.tracklet_frames.get(key, [])

        if len(all_frames) == 0:
            # Return a dummy single-frame tensor
            t = self.transform or get_transform(self.image_size, train=False)
            dummy = np.full((self.image_size, self.image_size, 3), 128, dtype=np.uint8)
            return t(dummy).unsqueeze(0), label

        exp_id = get_experiment_id(video)
        H = self._exp_resources[exp_id]["H"]

        # Sample frame indices
        sampled_indices = sample_frames(len(all_frames), self.num_frames)

        # Determine transform
        is_train = self.aug_config is not None
        if self.transform is not None:
            transform = self.transform
        else:
            transform = get_transform(self.image_size, train=is_train, aug_config=self.aug_config)

        # Seed RNG for consistent flip across all frames in this tracklet
        if is_train:
            rng_state = torch.random.get_rng_state()
            torch.manual_seed(idx)

        frame_tensors = []
        for si in sampled_indices:
            thermal_path, mask_path, rgb_frame_idx = all_frames[si]

            # Load thermal frame (grayscale)
            thermal_img = cv2.imread(thermal_path, cv2.IMREAD_GRAYSCALE)
            if thermal_img is None:
                thermal_img = np.full(self.thermal_shape, 128, dtype=np.uint8)

            # Load RGB mask
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                mask = (mask > 127).astype(np.uint8) * 255
            else:
                mask = np.ones(thermal_img.shape[:2], dtype=np.uint8) * 255

            # Warp mask from RGB space to thermal space
            warped_mask = warp_mask_to_thermal(mask, H, thermal_img.shape[:2])
            warped_mask_binary = (warped_mask > 127).astype(np.uint8)

            # Convert grayscale to 3-channel
            thermal_3ch = grayscale_to_3channel(thermal_img, self.colormap)

            # Apply mask and crop (reuse existing preprocessing)
            cropped = apply_mask_and_crop(thermal_3ch, warped_mask_binary, gray_fill=128)

            # No BGR->RGB conversion needed (already in RGB order from grayscale_to_3channel)

            # Seed RNG for consistent augmentation per frame within tracklet
            if is_train:
                torch.manual_seed(idx)

            frame_tensor = transform(cropped)
            frame_tensors.append(frame_tensor)

        # Restore RNG state
        if is_train:
            torch.random.set_rng_state(rng_state)

        frames = torch.stack(frame_tensors)  # (T, 3, 518, 518)
        return frames, label
