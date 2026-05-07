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

def split_video_into_scenes(input_path, output_dir, base_id, threshold=0.85):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    scene_idx = 0
    prev_hist = None
    frames_in_current_scene = 0
    min_scene_frames = int(fps)
    valid_scenes = []
    
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{base_id}-{scene_idx}.mp4")
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frames_in_current_scene += 1
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

        if prev_hist is not None:
            similarity = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            if similarity < threshold and frames_in_current_scene > min_scene_frames:
                out.release()
                valid_scenes.append(scene_idx)
                scene_idx += 1
                out_path = os.path.join(output_dir, f"{base_id}-{scene_idx}.mp4")
                out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
                frames_in_current_scene = 0
                
        out.write(frame)
        prev_hist = hist

    out.release()
    cap.release()
    
    if frames_in_current_scene > 0:
        if frames_in_current_scene < min_scene_frames and scene_idx > 0:
            os.remove(os.path.join(output_dir, f"{base_id}-{scene_idx}.mp4"))
        else:
            valid_scenes.append(scene_idx)
            
    return [f"{base_id}-{i}" for i in valid_scenes]

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

def render_video(video: Video, obj_frames, release_frame=-1, release_detector_name="", scene_num=""):
    orig_height = video.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    if orig_height > 0:
        target_scale = 720.0 / orig_height
    else:
        target_scale = 1.0
    video.scale = target_scale
    last_valid_frame = None
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
        hud = HUD(skeleton=skel, released=is_released, detector_name=release_detector_name)

        for obj in obj_frame:
            obj.draw(video, frame)

        hud.draw(video, frame)
        
        if scene_num:
            h, w = frame.shape[:2]
            ui_scale = h / 720.0
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8 * ui_scale
            thickness = max(1, int(2 * ui_scale))
            text = f"Scene {scene_num}"
            (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
            
            cv2.putText(
                frame,
                text,
                (w - tw - int(20 * ui_scale), h - int(20 * ui_scale)),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
            )

        video.write(frame)

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
        
        for i, obj_frame in enumerate(obj_frames):
            if release_frame != -1 and i > release_frame:
                break
                
            skel = next((obj for obj in obj_frame if isinstance(obj, Skeleton)), None)
            
            time_sec = i / fps if fps else 0
            released = release_frame != -1 and i >= release_frame
            
            active_side = "Unknown"
            if skel:
                if 15 in skel.landmarks and 16 in skel.landmarks:
                    if skel.landmarks[16].y < skel.landmarks[15].y:
                        active_side = "Right"
                    else:
                        active_side = "Left"
                        
            row = [i, f"{time_sec:.3f}", released, active_side, "", "", "", "", "", ""]
            
            if skel:
                row[4] = f"{skel.left_shoulder_angle:.1f}" if skel.left_shoulder_angle is not None else ""
                row[5] = f"{skel.left_elbow_angle:.1f}" if skel.left_elbow_angle is not None else ""
                row[6] = f"{skel.left_knee_angle:.1f}" if skel.left_knee_angle is not None else ""
                row[7] = f"{skel.right_shoulder_angle:.1f}" if skel.right_shoulder_angle is not None else ""
                row[8] = f"{skel.right_elbow_angle:.1f}" if skel.right_elbow_angle is not None else ""
                row[9] = f"{skel.right_knee_angle:.1f}" if skel.right_knee_angle is not None else ""
                
            writer.writerow(row)
