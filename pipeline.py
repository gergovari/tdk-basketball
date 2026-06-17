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

def extract_obj_frames(video: Video, yolo: YOLOFiltered, visualize=False):
    total_frames = len(video)
    obj_frames = [[] for x in range(total_frames)]
    for i, frame in enumerate(video):
        if i % 5 == 0 or i == total_frames - 1:
            print(f"\rExtracting frame {i+1}/{total_frames} ({(i+1)/total_frames*100:.1f}%)", end="", flush=True)
        obj_frames[i].extend(yolo.track(video, frame))
        if visualize:
            vis = frame.copy()
            for obj in obj_frames[i]:
                obj.draw(video, vis)
            cv2.imshow("YOLO Detection", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                visualize = False
                cv2.destroyWindow("YOLO Detection")
    print()
    if visualize:
        cv2.destroyWindow("YOLO Detection")
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

import math

def append_thrower_skeleton(video: Video, obj_frames, thrower_id, mediapipe, max_movement=60.0, visualize=False):
    last_valid_skeleton = None
    total_frames = len(video)
    
    # Calculate the "usual spot" by finding the median center of the thrower
    thrower_centers = []
    for frame_objs in obj_frames:
        for obj in frame_objs:
            if obj.id == thrower_id:
                cx = (obj.rect.x1 + obj.rect.x2) / 2
                cy = (obj.rect.y1 + obj.rect.y2) / 2
                thrower_centers.append((cx, cy))
                
    usual_spot = None
    if thrower_centers:
        thrower_centers.sort(key=lambda p: p[0])
        median_x = thrower_centers[len(thrower_centers)//2][0]
        thrower_centers.sort(key=lambda p: p[1])
        median_y = thrower_centers[len(thrower_centers)//2][1]
        usual_spot = (median_x, median_y)
        
    for i, frame in enumerate(video):
        if i % 5 == 0 or i == total_frames - 1:
            print(f"\rTracking skeleton {i+1}/{total_frames} ({(i+1)/total_frames*100:.1f}%)", end="", flush=True)
        show_this_frame = visualize  # track per-iteration so 'q' can disable
            
        current_thrower = next(
            (obj for obj in obj_frames[i] if obj.id == thrower_id), None
        )

        # Stop tracking if they move too far from the usual spot
        if current_thrower is not None and usual_spot is not None:
            cx = (current_thrower.rect.x1 + current_thrower.rect.x2) / 2
            cy = (current_thrower.rect.y1 + current_thrower.rect.y2) / 2
            dist_from_usual = math.hypot(cx - usual_spot[0], cy - usual_spot[1])
            if dist_from_usual > video.size[0] * 0.15:
                current_thrower = None
                last_valid_skeleton = None

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

                    is_valid = True
                    invalidation_reason = ""
                    if last_valid_skeleton is not None:
                        total_dist = 0
                        count = 0
                        for idx, lm in thrower_skeleton.landmarks.items():
                            if idx in last_valid_skeleton.landmarks:
                                old_lm = last_valid_skeleton.landmarks[idx]
                                total_dist += math.hypot(lm.x - old_lm.x, lm.y - old_lm.y)
                                count += 1
                        
                        if count > 0:
                            avg_dist = total_dist / count
                            
                            curr_xs = [lm.x for lm in thrower_skeleton.landmarks.values()]
                            curr_ys = [lm.y for lm in thrower_skeleton.landmarks.values()]
                            old_xs = [lm.x for lm in last_valid_skeleton.landmarks.values()]
                            old_ys = [lm.y for lm in last_valid_skeleton.landmarks.values()]
                            
                            if curr_xs and old_xs:
                                curr_width = max(curr_xs) - min(curr_xs)
                                curr_height = max(curr_ys) - min(curr_ys)
                                old_width = max(old_xs) - min(old_xs)
                                old_height = max(old_ys) - min(old_ys)
                                
                                if old_height > 0 and old_width > 0:
                                    curr_area = curr_width * curr_height
                                    old_area = old_width * old_height
                                    area_ratio = curr_area / old_area
                                    
                                    # 1. Scale-invariant Area Check (Resolution independent)
                                    if area_ratio > 1.8 or area_ratio < 0.55:
                                        is_valid = False
                                        invalidation_reason = f"Area jump ({area_ratio:.2f}x)"
                                        
                                    # 2. Dynamic Movement Threshold (scaled to the thrower's true pixel height)
                                    size_multiplier = old_height / 300.0
                                    dynamic_max_movement = max_movement * size_multiplier
                                    
                                    if is_valid and avg_dist > dynamic_max_movement:
                                        is_valid = False
                                        invalidation_reason = f"Moved {avg_dist:.1f}px (Limit: {dynamic_max_movement:.1f}px)"
                                else:
                                    if avg_dist > max_movement * video.scale:
                                        is_valid = False
                                        invalidation_reason = f"Moved {avg_dist:.1f}px (Fallback limit)"
                            else:
                                if avg_dist > max_movement * video.scale:
                                    is_valid = False
                                    invalidation_reason = f"Moved {avg_dist:.1f}px (Fallback limit)"
                        
                    if is_valid:
                        obj_frames[i].append(thrower_skeleton)
                        last_valid_skeleton = thrower_skeleton
                    else:
                        print(f"\nSkeleton correction applied at frame {i} ({invalidation_reason})")
                        cached_skel = Skeleton(landmarks=last_valid_skeleton.landmarks, detection_scale=last_valid_skeleton.detection_scale, is_cached=True)
                        obj_frames[i].append(cached_skel)
                elif last_valid_skeleton is not None:
                    # print(f"\nSkeleton correction applied at frame {i} (no landmarks detected)")
                    cached_skel = Skeleton(landmarks=last_valid_skeleton.landmarks, detection_scale=last_valid_skeleton.detection_scale, is_cached=True)
                    obj_frames[i].append(cached_skel)
        elif last_valid_skeleton is not None:
            # Uncomment the print below if debugging missing throwers
            # print(f"\nSkeleton correction applied at frame {i} (thrower missing)")
            cached_skel = Skeleton(landmarks=last_valid_skeleton.landmarks, detection_scale=last_valid_skeleton.detection_scale, is_cached=True)
            obj_frames[i].append(cached_skel)

        if show_this_frame:
            vis = frame.copy()
            for obj in obj_frames[i]:
                obj.draw(video, vis)
            cv2.imshow("Skeleton Tracking", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                visualize = False
                cv2.destroyWindow("Skeleton Tracking")

    print() # Newline after progress finishes
    if visualize:
        cv2.destroyWindow("Skeleton Tracking")
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

def render_video(video: Video, obj_frames, release_frame=-1, release_detector_name="", output_height=720.0):
    orig_height = video.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    if orig_height > 0:
        target_scale = output_height / orig_height
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

def render_throw_video(input_video_path, output_video_path, obj_frames, start_frame, end_frame, release_frame, fps=None, enable_hud=False, valid_frames=None, cycles=None, output_height=720.0):
    video = Video(input_video_path, output_video_path)
    orig_height = video.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    if orig_height > 0:
        target_scale = output_height / orig_height
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
        if i % 5 == 0 or i == total_frames - 1:
            print(f"\rRendering frame {i+1}/{total_frames} ({(i+1)/total_frames*100:.1f}%)", end="", flush=True)
            
        frame_idx = start_frame + i
        if release_frame != -1 and frame_idx > release_frame:
            frame_idx = release_frame
            
        if valid_frames is not None and frame_idx not in valid_frames:
            continue
            
        try:
            frame = video[frame_idx].copy()
            last_valid_frame = frame
        except IndexError:
            if last_valid_frame is not None:
                frame = last_valid_frame.copy()
            else:
                continue

        if enable_hud:
            skel = next((obj for obj in obj_frames[frame_idx] if isinstance(obj, Skeleton)), None)
            is_released = release_frame != -1 and frame_idx >= release_frame
            status_text = ""
            
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

            if cycles is not None:
                for prep, rel in cycles:
                    if prep <= frame_idx < rel:
                        status_text = "PREPARE"
                    elif frame_idx == rel:
                        status_text = "RELEASE"
            elif is_released:
                status_text = "RELEASED"
                
            hud = HUD(
                skeleton=skel, 
                status_text=status_text, 
                detector_name="SkeletonReleaseDetector" if status_text else "",
                active_side=last_active_side,
                angles=last_angles
            )

            for obj in obj_frames[frame_idx]:
                obj.draw(video, frame)

            hud.draw(video, frame)

        video.write(frame)

    print() # Newline after progress finishes
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
