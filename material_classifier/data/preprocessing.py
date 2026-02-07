import numpy as np
import torchvision.transforms as transforms


def apply_mask_and_crop(frame, mask, gray_fill=128):
    """
    Apply binary mask with gray fill and crop to bounding box.

    Args:
        frame: (H, W, 3) uint8 BGR or RGB image
        mask: (H, W) binary mask (bool or 0/1)
        gray_fill: fill value for non-object pixels (default 128)

    Returns:
        cropped: (h, w, 3) masked and cropped image, uint8
    """
    mask_float = mask.astype(np.float32)
    masked = frame * mask_float[..., None] + (1 - mask_float[..., None]) * gray_fill
    masked = masked.astype(np.uint8)

    rows, cols = np.where(mask > 0)
    if len(rows) == 0:
        return masked

    y1, y2 = rows.min(), rows.max() + 1
    x1, x2 = cols.min(), cols.max() + 1
    cropped = masked[y1:y2, x1:x2]
    return cropped


def get_transform(image_size=518, train=False, aug_config=None):
    """
    Build torchvision transform pipeline.

    Base: ToPILImage -> Resize(518,518) -> ToTensor -> Normalize(ImageNet)
    Train adds: RandomHorizontalFlip, ColorJitter (before ToTensor),
                RandomErasing (after ToTensor).
    """
    t_list = [transforms.ToPILImage()]

    if train and aug_config is not None:
        t_list.append(transforms.Resize((image_size, image_size)))
        if aug_config.get("random_horizontal_flip", False):
            t_list.append(transforms.RandomHorizontalFlip(p=0.5))
        cj = aug_config.get("color_jitter", {})
        if cj:
            t_list.append(transforms.ColorJitter(
                brightness=cj.get("brightness", 0),
                contrast=cj.get("contrast", 0),
                saturation=cj.get("saturation", 0),
                hue=cj.get("hue", 0),
            ))
        t_list.append(transforms.ToTensor())
        t_list.append(transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ))
        re_prob = aug_config.get("random_erasing", 0.0)
        if re_prob > 0:
            t_list.append(transforms.RandomErasing(p=re_prob))
    else:
        t_list.append(transforms.Resize((image_size, image_size)))
        t_list.append(transforms.ToTensor())
        t_list.append(transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ))

    return transforms.Compose(t_list)


def sample_frames(total_frames, num_frames=8):
    """
    Uniform temporal sampling of frame indices.

    If total_frames <= num_frames: return all indices.
    Else: return uniformly spaced indices.
    """
    if total_frames <= num_frames:
        return list(range(total_frames))
    return np.linspace(0, total_frames - 1, num_frames).astype(int).tolist()
