import cv2
from typing import List
from video import Video
from ml import YOLOFiltered
from entities import Skeleton, Landmark
from detectors import filter_obj_frame
from utils import ScaledInt
from ui import HUD

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

def append_thrower_skeleton(video: Video, obj_frames, thrower_id, mediapipe):
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
    earliest_detector = ""
    for d in detectors:
        idx = d.detect(obj_frames, fps)
        if idx != -1 and (earliest == -1 or idx < earliest):
            earliest = idx
            earliest_detector = d.__class__.__name__

    if earliest != -1:
        cut = obj_frames[: earliest + 1]
        last = list(cut[-1])

        for _ in range(3 * fps):
            cut.append(last)

        return cut, earliest, earliest_detector
    return obj_frames, -1, ""

def render_video(video: Video, obj_frames, release_frame=-1, release_detector_name=""):
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
        hud = HUD(skeleton=skel, released=is_released, detector_name=release_detector_name)

        for obj in obj_frame:
            obj.draw(video, frame)

        hud.draw(video, frame)

        video.write(frame)
