import cv2
import csv
from typing import List
from video import Video
from ml import YOLOFiltered
from entities import Skeleton, Landmark
from detectors import filter_obj_frame
from utils import ScaledInt
from ui import HUD
import os

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
    last_valid_frame = None
    
    last_active_side = "Unknown"
    last_angles = {"ls": None, "le": None, "lk": None, "rs": None, "re": None, "rk": None}
    
    for i, obj_frame in enumerate(obj_frames):
        frame_idx = min(i, len(video) - 1)
        if release_frame != -1:
            frame_idx = min(frame_idx, release_frame)

        try:
            frame = video[frame_idx].copy()
            last_valid_frame = frame
        except IndexError:
            if last_valid_frame is not None:
                frame = last_valid_frame.copy()
            else:
                continue

        skel = next((obj for obj in obj_frame if isinstance(obj, Skeleton)), None)
        is_released = release_frame != -1 and i >= release_frame
        
        if skel:
            if 15 in skel.landmarks and 16 in skel.landmarks:
                if skel.landmarks[16].y < skel.landmarks[15].y:
                    last_active_side = "Right"
                else:
                    last_active_side = "Left"
            if skel.left_shoulder_angle is not None: last_angles["ls"] = skel.left_shoulder_angle
            if skel.left_elbow_angle is not None: last_angles["le"] = skel.left_elbow_angle
            if skel.left_knee_angle is not None: last_angles["lk"] = skel.left_knee_angle
            if skel.right_shoulder_angle is not None: last_angles["rs"] = skel.right_shoulder_angle
            if skel.right_elbow_angle is not None: last_angles["re"] = skel.right_elbow_angle
            if skel.right_knee_angle is not None: last_angles["rk"] = skel.right_knee_angle
            
        hud = HUD(
            skeleton=skel, 
            released=is_released, 
            detector_name=release_detector_name,
            active_side=last_active_side,
            angles=last_angles
        )

        for obj in obj_frame:
            obj.draw(video, frame)

        hud.draw(video, frame)

        video.write(frame)

def render_throw_video(input_video_path, output_video_path, obj_frames, start_frame, end_frame, release_frame, release_detector_name="", fps=None):
    video = Video(input_video_path, output_video_path)
    orig_height = video.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    if orig_height > 0:
        target_scale = 720.0 / orig_height
    else:
        target_scale = 1.0
    video.scale = target_scale
    
    if fps is None:
        fps = video.fps

    last_valid_frame = None
    last_active_side = "Unknown"
    last_angles = {"ls": None, "le": None, "lk": None, "rs": None, "re": None, "rk": None}
    
    total_frames = (end_frame - start_frame + 1)
    
    for i in range(total_frames):
        frame_idx = start_frame + i
        if frame_idx > release_frame:
            frame_idx = release_frame
            
        try:
            frame = video[frame_idx].copy()
            last_valid_frame = frame
        except IndexError:
            if last_valid_frame is not None:
                frame = last_valid_frame.copy()
            else:
                continue

        obj_frame = obj_frames[frame_idx] if frame_idx < len(obj_frames) else []
        skel = next((obj for obj in obj_frame if isinstance(obj, Skeleton)), None)
        is_released = frame_idx >= release_frame
        
        if skel:
            if 15 in skel.landmarks and 16 in skel.landmarks:
                if skel.landmarks[16].y < skel.landmarks[15].y:
                    last_active_side = "Right"
                else:
                    last_active_side = "Left"
            if skel.left_shoulder_angle is not None: last_angles["ls"] = skel.left_shoulder_angle
            if skel.left_elbow_angle is not None: last_angles["le"] = skel.left_elbow_angle
            if skel.left_knee_angle is not None: last_angles["lk"] = skel.left_knee_angle
            if skel.right_shoulder_angle is not None: last_angles["rs"] = skel.right_shoulder_angle
            if skel.right_elbow_angle is not None: last_angles["re"] = skel.right_elbow_angle
            if skel.right_knee_angle is not None: last_angles["rk"] = skel.right_knee_angle
            
        hud = HUD(
            skeleton=skel, 
            released=is_released, 
            detector_name=release_detector_name,
            active_side=last_active_side,
            angles=last_angles
        )

        for obj in obj_frame:
            obj.draw(video, frame)

        hud.draw(video, frame)

        video.write(frame)

    video.release()

def export_skeleton_data(obj_frames, output_path, fps, release_frame):
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame", "time_sec", "released", "handedness",
            "left_shoulder", "left_elbow", "left_knee", 
            "right_shoulder", "right_elbow", "right_knee"
        ])
        
        last_active_side = "Unknown"
        last_angles = ["", "", "", "", "", ""]
        
        for i, obj_frame in enumerate(obj_frames):
            if release_frame != -1 and i > release_frame:
                break
                
            skel = next((obj for obj in obj_frame if isinstance(obj, Skeleton)), None)
            
            time_sec = i / fps if fps else 0
            released = release_frame != -1 and i >= release_frame
            
            if skel:
                if 15 in skel.landmarks and 16 in skel.landmarks:
                    if skel.landmarks[16].y < skel.landmarks[15].y:
                        last_active_side = "Right"
                    else:
                        last_active_side = "Left"
                        
            row = [i, f"{time_sec:.3f}", released, last_active_side, "", "", "", "", "", ""]
            
            if skel:
                row[4] = f"{skel.left_shoulder_angle:.1f}" if skel.left_shoulder_angle is not None else last_angles[0]
                row[5] = f"{skel.left_elbow_angle:.1f}" if skel.left_elbow_angle is not None else last_angles[1]
                row[6] = f"{skel.left_knee_angle:.1f}" if skel.left_knee_angle is not None else last_angles[2]
                row[7] = f"{skel.right_shoulder_angle:.1f}" if skel.right_shoulder_angle is not None else last_angles[3]
                row[8] = f"{skel.right_elbow_angle:.1f}" if skel.right_elbow_angle is not None else last_angles[4]
                row[9] = f"{skel.right_knee_angle:.1f}" if skel.right_knee_angle is not None else last_angles[5]
            else:
                row[4:10] = last_angles
                
            last_angles = row[4:10]
            writer.writerow(row)
