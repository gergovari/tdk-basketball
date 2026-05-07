from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
from typing import Dict
import cv2

from video import Video

class Drawable(ABC):
    @abstractmethod
    def draw(self, video: Video, frame):
        pass

@dataclass
class Rectangle(Drawable):
    x1: int
    y1: int
    x2: int
    y2: int
    detection_scale: float = 1.0

    @property
    def start_point(self):
        return (self.x1, self.y1)

    @property
    def end_point(self):
        return (self.x2, self.y2)

    @property
    def center(self):
        center_x = (self.x1 + self.x2) // 2
        center_y = (self.y1 + self.y2) // 2
        return (center_x, center_y)

    def distance_to(self, other_rect):
        return math.dist(self.center, other_rect.center)

    def with_padding(self, padding):
        return Rectangle(
            x1=self.x1 - padding,
            y1=self.y1 - padding,
            x2=self.x2 + padding,
            y2=self.y2 + padding,
            detection_scale=self.detection_scale,
        )

    def draw(self, video: Video, frame):
        ui_scale = frame.shape[0] / 720.0
        draw_ratio = video.scale / self.detection_scale
        pt1 = (int(self.x1 * draw_ratio), int(self.y1 * draw_ratio))
        pt2 = (int(self.x2 * draw_ratio), int(self.y2 * draw_ratio))
        thickness = max(1, int(1 * ui_scale))

        cv2.rectangle(
            frame,
            pt1,
            pt2,
            (0, 0, 255),
            thickness,
        )

@dataclass(eq=False)
class Object(Drawable):
    id: int
    conf: float
    name: str
    rect: Rectangle
    action: str = ""
    detection_scale: float = 1.0

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, Object):
            return False
        return self.id == other.id

    def draw_text(self, video: Video, frame):
        ui_scale = frame.shape[0] / 720.0
        draw_ratio = video.scale / self.detection_scale
        thickness = max(1, int(1 * ui_scale))
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.3 * ui_scale
        color = (147, 20, 255)

        text_x = int(self.rect.x1 * draw_ratio)
        text_y = int(self.rect.y1 * draw_ratio) - int(10 * ui_scale)

        cv2.putText(
            frame,
            f"[{self.id}]{self.name} ({(self.conf * 100):.1f}%)",
            (text_x, text_y),
            font,
            font_scale,
            color,
            thickness,
        )

        if self.action:
            (t_w, t_h), _ = cv2.getTextSize(self.action, font, font_scale, thickness)

            act_x = int(self.rect.x2 * draw_ratio) - t_w - int(2 * ui_scale)
            act_y = int(self.rect.y2 * draw_ratio) - int(2 * ui_scale)

            cv2.putText(
                frame,
                self.action,
                (act_x, act_y),
                font,
                font_scale,
                color,
                thickness,
            )

    def distance_to(self, other_obj):
        return self.rect.distance_to(other_obj.rect)

    def draw(self, video: Video, frame):
        self.rect.draw(video, frame)
        self.draw_text(video, frame)

@dataclass
class Landmark:
    x: int
    y: int
    visibility: float

POSE_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),
    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),
    (9, 10),
    (11, 12),
    (11, 13),
    (13, 15),
    (15, 17),
    (15, 19),
    (15, 21),
    (17, 19),
    (12, 14),
    (14, 16),
    (16, 18),
    (16, 20),
    (16, 22),
    (18, 20),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (24, 26),
    (25, 27),
    (26, 28),
    (27, 29),
    (28, 30),
    (29, 31),
    (30, 32),
    (27, 31),
    (28, 32),
]

@dataclass
class Skeleton(Drawable):
    landmarks: Dict[int, Landmark]
    visibility_threshold: float = 0
    detection_scale: float = 1.0

    def calculate_angle(self, a_idx, b_idx, c_idx):
        if (
            a_idx not in self.landmarks
            or b_idx not in self.landmarks
            or c_idx not in self.landmarks
        ):
            return None
        a = self.landmarks[a_idx]
        b = self.landmarks[b_idx]
        c = self.landmarks[c_idx]
        if (
            a.visibility <= self.visibility_threshold
            or b.visibility <= self.visibility_threshold
            or c.visibility <= self.visibility_threshold
        ):
            return None

        radians = math.atan2(c.y - b.y, c.x - b.x) - math.atan2(a.y - b.y, a.x - b.x)
        angle = abs(math.degrees(radians))
        return 360 - angle if angle > 180 else angle

    @property
    def left_elbow_angle(self):
        return self.calculate_angle(11, 13, 15)

    @property
    def right_elbow_angle(self):
        return self.calculate_angle(12, 14, 16)

    @property
    def left_shoulder_angle(self):
        return self.calculate_angle(23, 11, 13)

    @property
    def right_shoulder_angle(self):
        return self.calculate_angle(24, 12, 14)

    @property
    def left_knee_angle(self):
        return self.calculate_angle(23, 25, 27)

    @property
    def right_knee_angle(self):
        return self.calculate_angle(24, 26, 28)

    def draw(self, video: Video, frame):
        ui_scale = frame.shape[0] / 720.0
        draw_ratio = video.scale / self.detection_scale
        radius = max(2, int(2 * ui_scale))
        bone_thickness = max(1, int(1 * ui_scale))
        bone_color = (255, 255, 255)
        joint_color = (0, 255, 0)

        for connection in POSE_CONNECTIONS:
            idx1, idx2 = connection
            if idx1 in self.landmarks and idx2 in self.landmarks:
                lm1 = self.landmarks[idx1]
                lm2 = self.landmarks[idx2]

                if (
                    lm1.visibility > self.visibility_threshold
                    and lm2.visibility > self.visibility_threshold
                ):
                    pt1 = (int(lm1.x * draw_ratio), int(lm1.y * draw_ratio))
                    pt2 = (int(lm2.x * draw_ratio), int(lm2.y * draw_ratio))
                    cv2.line(frame, pt1, pt2, bone_color, bone_thickness)

        for idx, lm in self.landmarks.items():
            if lm.visibility > self.visibility_threshold:
                pt = (int(lm.x * draw_ratio), int(lm.y * draw_ratio))
                cv2.circle(frame, pt, radius, joint_color, -1)
