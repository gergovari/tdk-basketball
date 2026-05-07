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

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError("Video index out of range")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = self.cap.read()
        if not ret:
            raise IndexError("Failed to read video frame")
        return frame if self.scale == 1.0 else cv2.resize(frame, self.size)


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
        thrower_ids = {}
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
                    thrower_ids[closest_player] = None

        return list(thrower_ids.keys())


@dataclass
class ActionThrowerDetector(ThrowerDetector):
    action_filter: List[str] = field(default_factory=lambda: ["jump-shot"])

    def detect(self, obj_frames):
        throwers = {}

        for obj_frame in obj_frames:
            for obj in obj_frame:
                if self.action_filter and obj.action:
                    if any(word in obj.action for word in self.action_filter):
                        throwers[obj] = None

        return list(throwers.keys())


@dataclass
class CombinedThrowerDetector(ThrowerDetector):
    ball_filter: List[str] = field(default_factory=lambda: ["ball"])
    action_filter: List[str] = field(default_factory=lambda: ["jump-shot"])

    def __post_init__(self):
        self.ball_detector = BallProximityThrowerDetector(self.ball_filter)
        self.action_detector = ActionThrowerDetector(self.action_filter)

    def detect(self, obj_frames):
        all_balls = set()

        for obj_frame in obj_frames:
            for obj in obj_frame:
                if any(word in obj.name for word in self.ball_filter) and obj.id != -1:
                    all_balls.add(obj.id)

        shooters = self.action_detector.detect(obj_frames)
        players_with_ball = self.ball_detector.detect(obj_frames)

        num_balls = len(all_balls)
        valid_shooters = [s for s in shooters if s.id != -1]
        num_shooters = len(valid_shooters)

        if num_balls == 0 and num_shooters > 0:
            return [shooters[0]]
        elif num_balls == 1:
            overlap = [p for p in players_with_ball if p in shooters]
            return overlap if overlap else players_with_ball
        elif num_balls > 1 and num_shooters == 1:
            return shooters
        elif num_balls > 1 and num_shooters > 1:
            overlap = [p for p in players_with_ball if p in shooters]
            return (
                overlap
                if overlap
                else (players_with_ball if players_with_ball else shooters)
            )

        return []


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


class ReleaseDetector(ABC):
    @abstractmethod
    def detect(self, obj_frames, fps: int) -> int:
        pass


@dataclass
class ActionReleaseDetector(ReleaseDetector):
    def detect(self, obj_frames, fps: int) -> int:
        required_frames = fps * 0.2
        consecutive = 0
        for i, obj_frame in enumerate(obj_frames):
            has_jump_shot = any(
                getattr(obj, "action", "") == "jump-shot" for obj in obj_frame
            )
            if has_jump_shot:
                consecutive += 1
                if consecutive >= required_frames:
                    return i
            else:
                consecutive = 0
        return -1


@dataclass
class SkeletonReleaseDetector(ReleaseDetector):
    def detect(self, obj_frames, fps: int) -> int:
        for i, obj_frame in enumerate(obj_frames):
            for obj in obj_frame:
                if isinstance(obj, Skeleton):
                    la = (
                        obj.left_knee_angle,
                        obj.left_shoulder_angle,
                        obj.left_elbow_angle,
                    )
                    ra = (
                        obj.right_knee_angle,
                        obj.right_shoulder_angle,
                        obj.right_elbow_angle,
                    )

                    left_shot = (
                        all(x is not None for x in la)
                        and la[0] > 160
                        and la[1] > 120
                        and la[2] > 140
                    )
                    right_shot = (
                        all(x is not None for x in ra)
                        and ra[0] > 160
                        and ra[1] > 120
                        and ra[2] > 140
                    )

                    if left_shot or right_shot:
                        return i
        return -1


@dataclass
class HUD(Drawable):
    skeleton: Skeleton
    released: bool = False

    def draw(self, video: Video, frame):
        overlay = frame.copy()
        h, w = frame.shape[:2]
        ui_scale = h / 720.0
        box_w, box_h = int(250 * ui_scale), int(150 * ui_scale)
        box_x, box_y = w - box_w - int(20 * ui_scale), int(20 * ui_scale)

        cv2.rectangle(
            overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 0, 0), -1
        )
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5 * ui_scale
        thickness = max(1, int(1 * ui_scale))

        left_color = (0, 165, 255)  # Orange
        right_color = (255, 0, 0)  # Blue
        text_color = (255, 255, 255)

        y_offset = box_y + int(25 * ui_scale)

        active_side = "Unknown"
        if self.skeleton:
            if 15 in self.skeleton.landmarks and 16 in self.skeleton.landmarks:
                if self.skeleton.landmarks[16].y < self.skeleton.landmarks[15].y:
                    active_side = "Right"
                else:
                    active_side = "Left"

        cv2.putText(
            frame,
            f"Hand: {active_side}",
            (box_x + int(10 * ui_scale), y_offset),
            font,
            font_scale,
            text_color,
            thickness,
        )
        y_offset += int(25 * ui_scale)

        if self.skeleton:

            def draw_angle(name, left_val, right_val, y):
                l_str = f"{int(left_val)}" if left_val is not None else "N/A"
                r_str = f"{int(right_val)}" if right_val is not None else "N/A"
                cv2.putText(
                    frame,
                    f"{name}:",
                    (box_x + int(10 * ui_scale), y),
                    font,
                    font_scale,
                    text_color,
                    thickness,
                )
                cv2.putText(
                    frame,
                    l_str,
                    (box_x + int(120 * ui_scale), y),
                    font,
                    font_scale,
                    left_color,
                    thickness,
                )
                cv2.putText(
                    frame,
                    r_str,
                    (box_x + int(180 * ui_scale), y),
                    font,
                    font_scale,
                    right_color,
                    thickness,
                )

            draw_angle(
                "Shoulder",
                self.skeleton.left_shoulder_angle,
                self.skeleton.right_shoulder_angle,
                y_offset,
            )
            y_offset += int(25 * ui_scale)
            draw_angle(
                "Elbow",
                self.skeleton.left_elbow_angle,
                self.skeleton.right_elbow_angle,
                y_offset,
            )
            y_offset += int(25 * ui_scale)
            draw_angle(
                "Knee",
                self.skeleton.left_knee_angle,
                self.skeleton.right_knee_angle,
                y_offset,
            )

        if self.released:
            text = "RELEASED"
            (tw, th), _ = cv2.getTextSize(
                text,
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5 * ui_scale,
                max(2, int(3 * ui_scale)),
            )
            cv2.putText(
                frame,
                text,
                (int((w - tw) / 2), int((h + th) / 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5 * ui_scale,
                (0, 0, 255),
                max(2, int(3 * ui_scale)),
            )


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


def cut_after_release(obj_frames, detectors, fps):
    earliest = -1
    for d in detectors:
        idx = d.detect(obj_frames, fps)
        if idx != -1 and (earliest == -1 or idx < earliest):
            earliest = idx

    if earliest != -1:
        cut = obj_frames[: earliest + 1]
        last = list(cut[-1])

        for _ in range(3 * fps):
            cut.append(last)

        return cut, earliest
    return obj_frames, -1


def render_video(video: Video, obj_frames, release_frame=-1):
    orig_height = video.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    if orig_height > 0:
        target_scale = 720.0 / orig_height
    else:
        target_scale = 1.0
    video.scale = target_scale
    for i, obj_frame in enumerate(obj_frames):
        frame_idx = min(i, len(video) - 1)
        if release_frame != -1:
            frame_idx = min(frame_idx, release_frame)

        frame = video[frame_idx].copy()

        skel = next((obj for obj in obj_frame if isinstance(obj, Skeleton)), None)
        is_released = release_frame != -1 and i >= release_frame
        hud = HUD(skeleton=skel, released=is_released)

        for obj in obj_frame:
            obj.draw(video, frame)

        hud.draw(video, frame)

        video.write(frame)


params = InputParams(video_id="ft1_v108_002351_x264", data_path="data/")
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
detectors = [CombinedThrowerDetector(ball_filter=ball_filter)]
detected_throwers = list(
    dict.fromkeys(item for d in detectors for item in d.detect(obj_frames))
)
thrower_id = detected_throwers[0].id if detected_throwers else None
if thrower_id is not None:
    print(f"Thrower ID: {thrower_id}")
else:
    print("No valid thrower detected for this clip!")
print("Detected!")

print("Filter object frames...")
obj_frames = only_keep_relevant_obj_frames(obj_frames, ball_filter, thrower_id)
print("Filtered!")

print("Append skeleton of thrower...")
obj_frames = append_thrower_skeleton(video, obj_frames)
print("Tracked!")

print("Cut object frames after release...")
release_detectors = [ActionReleaseDetector(), SkeletonReleaseDetector()]
obj_frames, release_frame = cut_after_release(obj_frames, release_detectors, video.fps)
if release_frame == -1:
    print("Release not detected!")
print("Cut!")

print(f"Writing out to {params.output_video_path}...")
render_video(video, obj_frames, release_frame)
print("Wrote!")

video.release()

print("Done!")
