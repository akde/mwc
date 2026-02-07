import os
import numpy as np
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor


class Detectron2Detector:
    """Frozen Detectron2 Mask R-CNN inference wrapper."""

    def __init__(self, model_dir: str, confidence_threshold: float = 0.5, device: str = "cuda"):
        cfg = get_cfg()
        cfg.set_new_allowed(True)
        cfg.merge_from_file(os.path.join(model_dir, "config.yaml"))
        cfg.MODEL.WEIGHTS = os.path.join(model_dir, "model_best.pth")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
        cfg.MODEL.DEVICE = device
        self.predictor = DefaultPredictor(cfg)

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Run detection on a single BGR frame.

        Args:
            frame: BGR numpy array (H, W, 3), uint8

        Returns:
            List of dicts with 'bbox' [x1,y1,x2,y2], 'score' float, 'mask' (H,W) bool array.
        """
        outputs = self.predictor(frame)
        instances = outputs["instances"].to("cpu")

        results = []
        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()
        masks = instances.pred_masks.numpy()

        for i in range(len(instances)):
            results.append({
                "bbox": boxes[i].tolist(),
                "score": float(scores[i]),
                "mask": masks[i],  # (H, W) bool array
            })

        return results
