import csv
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from material_classifier.data.preprocessing import apply_mask_and_crop, get_transform, sample_frames

CLASS_NAMES = ["glass", "metal", "paper", "plastic"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}


class TrackletDataset(Dataset):
    """
    Dataset that loads raw frames + masks for each tracklet.
    Used during feature extraction (with DINOv2) and during full-pipeline training.
    """

    def __init__(self, labels_csv, tracklets_dir, videos_dir, split,
                 num_frames=8, transform=None, aug_config=None, image_size=518):
        self.tracklets_dir = tracklets_dir
        self.videos_dir = videos_dir
        self.num_frames = num_frames
        self.transform = transform
        self.aug_config = aug_config
        self.image_size = image_size

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

        # Build per-tracklet frame index: {(video, track_id): [(frame_idx, mask_path), ...]}
        self.tracklet_frames = {}
        for video, track_id, _ in self.samples:
            key = (video, track_id)
            if key in self.tracklet_frames:
                continue

            video_stem = os.path.splitext(video)[0]
            csv_path = os.path.join(tracklets_dir, video_stem, "tracklet_data.csv")
            masks_dir = os.path.join(tracklets_dir, video_stem, "masks")

            if not os.path.exists(csv_path):
                self.tracklet_frames[key] = []
                continue

            frames = []
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if int(row["track_id"]) == track_id:
                        frame_idx = int(row["frame"])
                        mask_path = os.path.join(
                            masks_dir,
                            f"frame_{frame_idx:06d}_track_{track_id}.png"
                        )
                        frames.append((frame_idx, mask_path))

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

        video_path = os.path.join(self.videos_dir, video)
        cap = cv2.VideoCapture(video_path)

        frame_tensors = []
        for si in sampled_indices:
            frame_idx, mask_path = all_frames[si]

            # Read frame from video
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                # Fallback: gray frame
                frame = np.full((int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                                 int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 3),
                                128, dtype=np.uint8)

            # Load mask
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                mask = (mask > 127).astype(np.uint8)
            else:
                mask = np.ones(frame.shape[:2], dtype=np.uint8)

            # Apply mask and crop
            cropped = apply_mask_and_crop(frame, mask, gray_fill=128)

            # BGR -> RGB
            rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

            # Seed RNG for consistent augmentation per frame within tracklet
            if is_train:
                torch.manual_seed(idx)

            frame_tensor = transform(rgb)
            frame_tensors.append(frame_tensor)

        cap.release()

        # Restore RNG state
        if is_train:
            torch.random.set_rng_state(rng_state)

        frames = torch.stack(frame_tensors)  # (T, 3, 518, 518)
        return frames, label


class CachedFeatureDataset(Dataset):
    """
    Dataset that loads precomputed DINOv2 [CLS] features from disk.
    Used during training (no need to load DINOv2 backbone).
    """

    def __init__(self, labels_csv, features_dir, split):
        self.features_dir = features_dir
        self.samples = []  # (feature_path, label)

        with open(labels_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] == split:
                    video_stem = os.path.splitext(row["video"])[0]
                    feature_path = os.path.join(
                        features_dir,
                        f"{video_stem}_track_{row['track_id']}.pt"
                    )
                    self.samples.append((
                        feature_path,
                        CLASS_TO_IDX[row["class"]],
                    ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feature_path, label = self.samples[idx]
        features = torch.load(feature_path, weights_only=True)  # (T, 1024)
        return features, label


def collate_fn(batch):
    """
    Custom collate for variable-length sequences.
    Pads all sequences to max length in batch, creates boolean mask.

    Returns:
        padded: (B, max_T, ...) — either (B, max_T, 3, 518, 518) or (B, max_T, 1024)
        masks: (B, max_T) boolean, True = valid frame
        labels: (B,) long tensor
    """
    max_T = max(f.shape[0] for f, _ in batch)
    B = len(batch)
    first_shape = batch[0][0].shape[1:]  # e.g., (3, 518, 518) or (1024,)
    padded = torch.zeros(B, max_T, *first_shape)
    masks = torch.zeros(B, max_T, dtype=torch.bool)
    labels = torch.zeros(B, dtype=torch.long)
    for i, (f, l) in enumerate(batch):
        T_i = f.shape[0]
        padded[i, :T_i] = f
        masks[i, :T_i] = True
        labels[i] = l
    return padded, masks, labels
