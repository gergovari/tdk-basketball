from dataclasses import dataclass
from typing import List, Tuple
import cv2
from ultralytics import YOLO

from entities import Rectangle, Object, Skeleton, Landmark


# COCO 17 keypoints → pipeline landmark indices
# We keep the same index mapping used by the rest of the pipeline
# (originally derived from MediaPipe's 33-landmark layout, but only
# the 17 COCO-mapped indices are populated — which is all the pipeline
# ever actually uses: shoulders, elbows, wrists, hips, knees, ankles).
_COCO_TO_PIPELINE = {
    0: 0,   1: 2,   2: 5,   3: 7,   4: 8,     # nose, eyes, ears
    5: 11,  6: 12,  7: 13,  8: 14,  9: 15, 10: 16,  # shoulders, elbows, wrists
    11: 23, 12: 24, 13: 25, 14: 26, 15: 27, 16: 28,  # hips, knees, ankles
}



class YOLOPose:
    """Unified person detector + pose estimator using YOLO-Pose.

    A single model call detects people AND estimates their keypoints.
    Returns both Object (bounding box) and Skeleton entities so the
    pipeline no longer needs a separate YOLO detection model.
    """

    def __init__(self, model_path='models/yolov8x-pose-p6.engine'):
        import os
        import torch

        # Auto-compile TensorRT engine from .pt weights if the .engine doesn't exist
        if model_path.endswith('.engine') and not os.path.exists(model_path):
            import subprocess, sys
            pt_path = model_path.replace('.engine', '.pt')
            if not os.path.exists(pt_path):
                raise FileNotFoundError(
                    f"Neither '{model_path}' nor '{pt_path}' found. "
                    "Place the .pt weights in the models/ directory."
                )
            print(f"[ml] TensorRT engine not found. Compiling from {pt_path} ...")
            print(f"[ml] Running compilation in isolated subprocess to limit RAM usage ...")
            ret = subprocess.run(
                [sys.executable, "compile_trt.py"],
                cwd=os.path.dirname(os.path.abspath(__file__)),
            )
            if ret.returncode != 0 or not os.path.exists(model_path):
                raise RuntimeError(
                    f"TensorRT engine compilation failed (exit code {ret.returncode}).\n"
                    "This is likely an OOM kill — the yolov8x-pose-p6 export needs ~12GB RAM.\n"
                    "To fix:\n"
                    "  1. Close browsers, IDEs, and other heavy apps\n"
                    "  2. Optionally add temporary swap:  sudo fallocate -l 16G /swapfile_temp && sudo chmod 600 /swapfile_temp && sudo mkswap /swapfile_temp && sudo swapon /swapfile_temp\n"
                    "  3. Run:  python compile_trt.py\n"
                    "  4. Then re-run the pipeline."
                )
            print(f"[ml] Compilation complete → {model_path}")

        self.model = YOLO(model_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.use_half = torch.cuda.is_available()
        
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            
            # Warmup inference to force TensorRT to initialize its CUDA context
            # This MUST happen before OpenCV initializes hardware video decoding
            # otherwise they conflict and trigger CUDA Error 100.
            import numpy as np
            dummy_frame = np.zeros((320, 320, 3), dtype=np.uint8)
            self.model(dummy_frame, verbose=False, imgsz=320, device=self.device, half=self.use_half)

    def reset(self):
        """Reset tracker state between videos."""
        self.model.predictor = None

    def track(self, video, frame, imgsz=320) -> List[Tuple[Object, Skeleton]]:
        """Run pose tracking on a full frame.

        Returns a list of (Object, Skeleton) pairs — one per detected person.
        The Object carries the bounding box and tracking ID.
        The Skeleton carries the keypoints in full-frame pixel coordinates.
        """
        results = self.model.track(
            frame, persist=True, verbose=False,
            imgsz=imgsz, half=self.use_half, device=self.device,
            tracker="bytetrack.yaml", classes=[0]
        )
        result = results[0]

        if result.boxes is None or len(result.boxes) == 0:
            return []

        has_keypoints = (
            result.keypoints is not None
            and len(result.keypoints) > 0
            and result.keypoints.conf is not None
        )

        detections = []
        
        # Batch copy all tensors to CPU/NumPy to prevent hundreds of tiny PCIe syncs per frame
        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        boxes_conf = result.boxes.conf.cpu().numpy()
        boxes_id = result.boxes.id.cpu().numpy() if result.boxes.id is not None else None
        
        if has_keypoints:
            kps_xy = result.keypoints.xy.cpu().numpy()
            kps_conf = result.keypoints.conf.cpu().numpy()

        for det_idx in range(len(boxes_xyxy)):
            xyxy = boxes_xyxy[det_idx]
            rect = Rectangle(*map(int, xyxy), detection_scale=video.scale)
            conf = float(boxes_conf[det_idx])

            if boxes_id is not None:
                track_id = int(boxes_id[det_idx])
            else:
                track_id = -1

            obj = Object(
                name="person",
                rect=rect,
                id=track_id,
                conf=conf,
                detection_scale=video.scale,
            )

            skeleton = None
            if has_keypoints:
                xy = kps_xy[det_idx]    # [17, 2] pixel coords
                kp_conf = kps_conf[det_idx]  # [17]

                extracted_landmarks = {}
                for coco_i, pipe_i in _COCO_TO_PIPELINE.items():
                    v = float(kp_conf[coco_i])
                    if v > 0:
                        extracted_landmarks[pipe_i] = Landmark(
                            x=int(xy[coco_i][0]),
                            y=int(xy[coco_i][1]),
                            visibility=v,
                        )

                if extracted_landmarks:
                    skeleton = Skeleton(
                        landmarks=extracted_landmarks,
                        detection_scale=video.scale,
                    )

            detections.append((obj, skeleton))

        return detections

    def detect_on_crop(self, frame, rect, video_scale) -> Skeleton:
        """Run pose estimation on a crop region and return a Skeleton in full-frame coordinates.

        Used as a fallback when the tracker loses someone but we still have
        the last known bounding box.
        """
        crop = frame[max(0, rect.y1):rect.y2, max(0, rect.x1):rect.x2]
        crop_h, crop_w = crop.shape[:2]
        if crop_h <= 0 or crop_w <= 0:
            return None

        # Downscale large crops for speed
        max_crop_h = 480
        if crop_h > max_crop_h:
            scale_factor = max_crop_h / crop_h
            small_crop = cv2.resize(crop, (int(crop_w * scale_factor), max_crop_h))
            small_h, small_w = small_crop.shape[:2]
        else:
            small_crop = crop
            small_h, small_w = crop_h, crop_w
            scale_factor = 1.0

        results = self.model(small_crop, verbose=False, device=self.device, half=self.use_half)
        result = results[0]

        if (result.keypoints is None or len(result.keypoints) == 0
                or result.keypoints.conf is None):
            return None

        # Pick the most confident detection (crop should contain one person)
        best = 0
        if result.boxes is not None and len(result.boxes) > 1:
            best = int(result.boxes.conf.argmax())

        xy = result.keypoints.xy[best]    # [17, 2] pixel coords in the small_crop
        conf = result.keypoints.conf[best]  # [17]

        extracted_landmarks = {}
        for coco_i, pipe_i in _COCO_TO_PIPELINE.items():
            v = float(conf[coco_i])
            if v > 0:
                # Map back from small_crop → original crop → full frame
                crop_x_px = float(xy[coco_i][0]) / scale_factor
                crop_y_px = float(xy[coco_i][1]) / scale_factor
                full_x_px = int(crop_x_px + rect.x1)
                full_y_px = int(crop_y_px + rect.y1)

                extracted_landmarks[pipe_i] = Landmark(
                    x=full_x_px, y=full_y_px, visibility=v,
                )

        if not extracted_landmarks:
            return None

        return Skeleton(
            landmarks=extracted_landmarks,
            detection_scale=video_scale,
        )
