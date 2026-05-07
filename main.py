from ultralytics import YOLO
import cv2
from dataclasses import dataclass, field
import math
from abc import ABC, abstractmethod
import os
from typing import Dict, Tuple, List
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


@dataclass
class ScaledInt:
    Value: int
    ref_res: Tuple[int, int]
    res: Tuple[int, int]

    @property
    def value(self) -> int:
        scale_factor = self.res[0] / self.ref_res[0]
        return int(self.Value * scale_factor)

    @value.setter
    def value(self, new_val: int):
        scale_factor = self.res[0] / self.ref_res[0]
        self.Value = int(new_val / scale_factor)


class Video:
    def __init__(self, input_video_path, output_video_path, scale=1.0):
        self.Scale = scale
        self.output_video_path = output_video_path
        self.cap = cv2.VideoCapture(input_video_path)
        self.out = cv2.VideoWriter(
            self.output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, self.size
        )
        self.frames_written = 0

    @property
    def size(self):
        width = int(int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) * self.scale)
        height = int(int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) * self.scale)
        return (width, height)

    @property
    def scale(self):
        return self.Scale

    @scale.setter
    def scale(self, new_scale):
        if hasattr(self, "out"):
            self.out.release()

        temp_old_video = self.output_video_path + ".old.mp4"
        has_valid_old_video = False

        if os.path.exists(self.output_video_path):
            if self.frames_written > 0:
                os.rename(self.output_video_path, temp_old_video)
                has_valid_old_video = True
            else:
                os.remove(self.output_video_path)

        self.Scale = new_scale

        self.out = cv2.VideoWriter(
            self.output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, self.size
        )

        if has_valid_old_video and os.path.exists(temp_old_video):
            old_cap = cv2.VideoCapture(temp_old_video)

            while old_cap.isOpened():
                ret, frame = old_cap.read()
                if not ret:
                    break

                resized_frame = cv2.resize(frame, self.size)
                self.out.write(resized_frame)

            old_cap.release()
            os.remove(temp_old_video)

    def __len__(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def fps(self):
        return int(self.cap.get(cv2.CAP_PROP_FPS))

    def write(self, frame):
        self.out.write(frame)
        self.frames_written += 1

    def release(self):
        if self.cap.isOpened():
            self.cap.release()
        if hasattr(self, "out"):
            self.out.release()

    def __del__(self):
        self.release()

    def __iter__(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return self

    def __next__(self):
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return frame if self.scale == 1.0 else cv2.resize(frame, self.size)
        raise StopIteration


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

    def draw(self, video, frame):
        draw_ratio = video.scale / self.detection_scale
        pt1 = (int(self.x1 * draw_ratio), int(self.y1 * draw_ratio))
        pt2 = (int(self.x2 * draw_ratio), int(self.y2 * draw_ratio))
        thickness = max(1, int(1 * video.scale))

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
        draw_ratio = video.scale / self.detection_scale
        thickness = max(1, int(1 * video.scale))
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.3 * video.scale
        color = (147, 20, 255)

        text_x = int(self.rect.x1 * draw_ratio)
        text_y = int(self.rect.y1 * draw_ratio) - int(10 * video.scale)

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

            act_x = int(self.rect.x2 * draw_ratio) - t_w - int(2 * video.scale)
            act_y = int(self.rect.y2 * draw_ratio) - int(2 * video.scale)

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

    def draw(self, video, frame):
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

    def draw(self, video: Video, frame):
        draw_ratio = video.scale / self.detection_scale
        radius = max(2, int(2 * video.scale))
        bone_thickness = max(1, int(1 * video.scale))
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


@dataclass
class YOLOFiltered:
    model: YOLO
    name_filter: List

    def track(self, video, frame):
        objects = []
        results = self.model.track(frame, persist=True, verbose=False)
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


@dataclass
class InputParams:
    video_id: str
    data_path: str

    @property
    def input_video_path(self):
        return f"{self.data_path}/input/{self.video_id}.mp4"

    @property
    def output_video_path(self):
        return f"{self.data_path}/output/{self.video_id}.mp4"

    @property
    def output_data_path(self):
        return f"{self.data_path}/data/{self.video_id}.json"


class ThrowerDetector(ABC):
    @abstractmethod
    def detect(self, obj_frames):
        pass


def filter_obj_frame(func, obj_frame):
    return list(
        filter(
            func,
            obj_frame,
        )
    )


@dataclass
class BallProximityThrowerDetector(ThrowerDetector):
    ball_filter: List[str] = field(default_factory=lambda: ["ball"])

    def detect(self, obj_frames):
        thrower_ids = set()
        relevant_obj_frames = []
        for obj_frame in obj_frames:
            for obj in obj_frame:
                if any(word in obj.name for word in self.ball_filter):
                    relevant_obj_frames.append(obj_frame)
                    break

        for obj_frame in relevant_obj_frames:
            balls = filter_obj_frame(
                lambda x: any(word in x.name for word in self.ball_filter), obj_frame
            )
            players = filter_obj_frame(
                lambda x: not any(word in x.name for word in self.ball_filter),
                obj_frame,
            )

            for ball in balls:
                smallest_dist = float("inf")
                closest_player = None
                for player in players:
                    dist = player.distance_to(ball)
                    if dist < smallest_dist:
                        smallest_dist = dist
                        closest_player = player

                if closest_player is not None:
                    thrower_ids.add(closest_player)

        return thrower_ids


@dataclass
class ActionThrowerDetector(ThrowerDetector):
    action_filter: List[str] = field(default_factory=lambda: ["jump-shot"])

    def detect(self, obj_frames):
        throwers = set()

        for obj_frame in obj_frames:
            for obj in obj_frame:
                if self.action_filter and obj.action:
                    if any(word in obj.action for word in self.action_filter):
                        throwers.add(obj)

        return throwers


@dataclass
class YOLOParams:
    model_path: str
    name_filter: List[str]


@dataclass
class MediaPipeParams:
    model_path: str
    min_pose_conf: float
    min_track_conf: float


class MediaPipe:
    def __init__(self, mp_params):
        base_options = python.BaseOptions(model_asset_path=mp_params.model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False,
            min_pose_detection_confidence=mp_params.min_pose_conf,
            min_tracking_confidence=mp_params.min_track_conf,
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)

    def detect(self, frame):
        crop_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)
        return self.detector.detect(mp_image)


def extract_obj_frames(video: Video, yolo: YOLOFiltered):
    obj_frames = [[] for x in range(len(video))]
    for i, frame in enumerate(video):
        obj_frames[i].extend(yolo.track(video, frame))
    return obj_frames


def enrich_player_with_action(player_filter: List[str], obj_frames):
    for obj_frame in obj_frames:
        for obj in obj_frame:
            if any(word in obj.name for word in player_filter):
                if "-" in obj.name:
                    obj.name, obj.action = obj.name.split("-", 1)
    return obj_frames


def only_keep_relevant_obj_frames(obj_frames, ball_filter, thrower_id):
    filtered_obj_frames = []
    for obj_frame in obj_frames:
        filtered_frame = filter_obj_frame(
            lambda x: any(word in x.name for word in ball_filter) or x.id == thrower_id,
            obj_frame,
        )
        filtered_obj_frames.append(filtered_frame)
    return filtered_obj_frames


def append_thrower_skeleton(video: Video, obj_frames):
    for i, frame in enumerate(video):
        current_thrower = next(
            (obj for obj in obj_frames[i] if obj.id == thrower_id), None
        )

        if current_thrower is not None:
            rect = current_thrower.rect.with_padding(
                ScaledInt(60, (1280, 720), video.size).value
            )
            thrower_crop = frame[max(0, rect.y1) : rect.y2, max(0, rect.x1) : rect.x2]

            crop_h, crop_w = thrower_crop.shape[:2]
            if crop_h > 0 and crop_w > 0:
                detection_result = mediapipe.detect(thrower_crop)

                if detection_result.pose_landmarks:
                    pose_landmarks = detection_result.pose_landmarks[0]
                    extracted_landmarks = {}

                    for idx, lm in enumerate(pose_landmarks):
                        crop_x_px = lm.x * crop_w
                        crop_y_px = lm.y * crop_h
                        full_x_px = int(crop_x_px + rect.x1)
                        full_y_px = int(crop_y_px + rect.y1)

                        extracted_landmarks[idx] = Landmark(
                            x=full_x_px, y=full_y_px, visibility=lm.visibility
                        )

                    thrower_skeleton = Skeleton(
                        landmarks=extracted_landmarks,
                        detection_scale=video.scale,
                    )
                    obj_frames[i].append(thrower_skeleton)
    return obj_frames


def render_video(video: Video, obj_frames):
    orig_height = video.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    if orig_height > 0:
        target_scale = 720.0 / orig_height
    else:
        target_scale = 1.0
    video.scale = target_scale
    for i, frame in enumerate(video):
        objects = obj_frames[i]
        for obj in objects:
            obj.draw(video, frame)

        video.write(frame)
    video.scale = 1


params = InputParams(video_id="nba1", data_path="data/")
player_filter = ["player", "person", "human"]
ball_filter = ["ball"]
yolo_filter = player_filter + ball_filter
yolo_params = YOLOParams(model_path="models/basketball-3-m.pt", name_filter=yolo_filter)
mp_params = MediaPipeParams(
    model_path="models/pose_landmarker.task", min_pose_conf=0.1, min_track_conf=0.1
)

model = YOLO(yolo_params.model_path)
yolo_filtered = YOLOFiltered(model, yolo_params.name_filter)

mediapipe = MediaPipe(mp_params)

video = Video(params.input_video_path, params.output_video_path)

print(f"Extracting object frames from {params.input_video_path}...")
obj_frames = extract_obj_frames(video, yolo_filtered)
print("Extracted!")

print("Enrich player data with action...")
obj_frames = enrich_player_with_action(player_filter, obj_frames)
print("Enriched!")

print("Detect thrower...")
detectors = [ActionThrowerDetector(), BallProximityThrowerDetector(ball_filter)]
detected_throwers = list(
    dict.fromkeys(item for d in detectors for item in d.detect(obj_frames))
)
thrower_id = detected_throwers[0].id
print("Detected!")

print("Filter object frames...")
obj_frames = only_keep_relevant_obj_frames(obj_frames, ball_filter, thrower_id)
print("Filtered!")

print("Append skeleton of thrower...")
obj_frames = append_thrower_skeleton(video, obj_frames)
print("Tracked!")

print(f"Writing out to {params.output_video_path}...")
render_video(video, obj_frames)
print("Wrote!")

video.release()

print("Done!")
