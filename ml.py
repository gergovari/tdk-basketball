from dataclasses import dataclass
from typing import List
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO

from entities import Rectangle, Object

@dataclass
class YOLOFiltered:
    model: YOLO
    name_filter: List
    imgsz: int = 320

    def __post_init__(self):
        import torch
        self.use_half = torch.cuda.is_available()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def track(self, video, frame):
        objects = []
        results = self.model.track(frame, persist=True, verbose=False, imgsz=self.imgsz, half=self.use_half, device=self.device)
        result = results[0]

        for box in result.boxes:
            rect = Rectangle(*map(int, box.xyxy[0]), detection_scale=video.scale)
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            name = self.model.names[cls_id]

            if any(word in name for word in self.name_filter):
                if box.id is not None:
                    track_id = int(box.id[0])
                else:
                    track_id = -1

                obj = Object(
                    name=name,
                    rect=rect,
                    id=track_id,
                    conf=conf,
                    detection_scale=video.scale,
                )
                objects.append(obj)
        return objects

class MediaPipe:
    def __init__(self, mp_params):
        self._mp_params = mp_params
        base_options = python.BaseOptions(model_asset_path=mp_params.model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False,
            min_pose_detection_confidence=mp_params.min_pose_conf,
            min_tracking_confidence=mp_params.min_track_conf,
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)

    def reset(self):
        """No-op for IMAGE mode (kept for API compatibility)."""
        pass

    def detect(self, frame, timestamp_ms=None):
        crop_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)
        return self.detector.detect(mp_image)


# --- YOLO-Pose GPU backend (drop-in replacement for MediaPipe) ---

# COCO 17 keypoints → MediaPipe 33 landmark indices
_COCO_TO_MP = {
    0: 0,   1: 2,   2: 5,   3: 7,   4: 8,     # nose, eyes, ears
    5: 11,  6: 12,  7: 13,  8: 14,  9: 15, 10: 16,  # shoulders, elbows, wrists
    11: 23, 12: 24, 13: 25, 14: 26, 15: 27, 16: 28,  # hips, knees, ankles
}

class _Lm:
    """Lightweight landmark matching MediaPipe's interface (x/y normalized, visibility)."""
    __slots__ = ('x', 'y', 'visibility')
    def __init__(self, x, y, visibility):
        self.x = x; self.y = y; self.visibility = visibility

class _PoseResult:
    """Mimics MediaPipe's PoseLandmarkerResult.pose_landmarks."""
    __slots__ = ('pose_landmarks',)
    def __init__(self, landmarks_list):
        self.pose_landmarks = landmarks_list

class YOLOPose:
    """GPU-accelerated pose estimator using YOLO-Pose. Same detect() interface as MediaPipe."""

    def __init__(self, model_path='models/yolo11n-pose.pt'):
        import torch
        self.model = YOLO(model_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.use_half = torch.cuda.is_available()

    def reset(self):
        pass

    def detect(self, frame, timestamp_ms=None):
        results = self.model(frame, verbose=False, device=self.device, half=self.use_half)
        result = results[0]

        if result.keypoints is None or len(result.keypoints) == 0 or result.keypoints.conf is None:
            return _PoseResult([])

        # Pick the most confident detection (crop should contain one person)
        best = 0
        if result.boxes is not None and len(result.boxes) > 1:
            best = int(result.boxes.conf.argmax())

        xy = result.keypoints.xyn[best]   # [17, 2] normalized
        conf = result.keypoints.conf[best] # [17]

        # Build a list matching MediaPipe's 29-element layout (indices 0-28)
        # None entries are skipped by the pipeline
        mp_lms = [None] * 29
        for coco_i, mp_i in _COCO_TO_MP.items():
            mp_lms[mp_i] = _Lm(float(xy[coco_i][0]), float(xy[coco_i][1]), float(conf[coco_i]))

        return _PoseResult([mp_lms])

