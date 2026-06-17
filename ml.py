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

