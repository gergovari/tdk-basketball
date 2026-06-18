import cv2
import csv
from typing import List
from video import Video
from ml import YOLOPose
from entities import Skeleton, Landmark
from detectors import filter_obj_frame
from utils import ScaledInt
from ui import HUD
import os
import math
import time
import sys
import select
try:
    import termios
    import tty
except ImportError:
    pass


def extract_and_refine_obj_frames(video: Video, yolo_pose: YOLOPose, max_movement=60.0, visualize=False, enable_invalidation=False, min_kp_conf=0.3, min_keypoints=6, lowpass=0.4):
    """Run YOLOPose on the video, extract tracking data, and apply refinement filters in a single pass.
    
    This replaces separate extraction and refinement steps by processing the video
    sequentially, removing the need for slow seek operations. All tracked skeletons
    are refined on-the-fly.
    """
    total_frames = len(video)
    obj_frames = [[] for _ in range(total_frames)]
    
    t_start = time.perf_counter()
    window_created = False
    video.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    last_valid_skeletons = {}
    last_smoothed_skeletons = {}
    last_known_rects = {}
    
    filter_desc = []
    if min_kp_conf > 0: filter_desc.append(f"kp_conf>={min_kp_conf}")
    if min_keypoints > 0: filter_desc.append(f"min_kp>={min_keypoints}")
    if enable_invalidation: filter_desc.append(f"inval(max_move={max_movement})")
    if lowpass > 0: filter_desc.append(f"lowpass(a={lowpass})")
    print(f"  Filters: {', '.join(filter_desc) if filter_desc else 'none'}")
    
    has_termios = 'termios' in sys.modules
    if has_termios:
        try:
            import atexit
            old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
            atexit.register(termios.tcsetattr, sys.stdin, termios.TCSADRAIN, old_settings)
        except Exception:
            has_termios = False

    for i in range(total_frames):
        if i % 5 == 0 or i == total_frames - 1:
            elapsed = time.perf_counter() - t_start
            fps = (i + 1) / elapsed if elapsed > 0 else 0
            if fps > 0:
                eta_s = (total_frames - (i + 1)) / fps
                m, s = divmod(int(eta_s), 60)
                eta_str = f" | ETA: {m}m {s:02d}s"
            else:
                eta_str = ""
            print(f"\rExtracting & Refining {i+1}/{total_frames} ({(i+1)/total_frames*100:.1f}%) — {fps:.1f} fps{eta_str}", end="", flush=True)
            
        ret, frame = video.cap.read()
        if not ret: break
        
        detections = yolo_pose.track(video, frame)
        active_tracks = set()
        
        # 1. First loop: handle detections found by standard tracking
        for obj, skel in detections:
            active_tracks.add(obj.id)
            obj_frames[i].append(obj)
            last_known_rects[obj.id] = obj.rect.with_padding(ScaledInt(60, (1280, 720), video.size).value)
            if skel is not None:
                skel._track_id = obj.id
                
        # 2. Handle missing tracks (dropout fallback)
        for track_id, rect in list(last_known_rects.items()):
            if track_id not in active_tracks:
                fallback_skel = yolo_pose.detect_on_crop(frame, rect, video.scale)
                if fallback_skel is not None:
                    fallback_skel._track_id = track_id
                    detections.append((None, fallback_skel))
                    
        # 3. Refine all skeletons for this frame
        for _, skel in detections:
            if skel is None:
                continue
                
            track_id = skel._track_id
            
            if min_kp_conf > 0:
                skel = _apply_confidence_filter(skel, min_kp_conf)
                
            if min_keypoints > 0 and len(skel.landmarks) < min_keypoints:
                skel = None
                
            if skel is not None:
                is_valid = True
                invalidation_reason = ""
                last_valid = last_valid_skeletons.get(track_id)
                
                if enable_invalidation and last_valid is not None:
                    total_dist = 0
                    count = 0
                    for idx, lm in skel.landmarks.items():
                        if idx in last_valid.landmarks:
                            old_lm = last_valid.landmarks[idx]
                            total_dist += math.hypot(lm.x - old_lm.x, lm.y - old_lm.y)
                            count += 1
                            
                    if count > 0:
                        avg_dist = total_dist / count
                        curr_xs = [lm.x for lm in skel.landmarks.values()]
                        curr_ys = [lm.y for lm in skel.landmarks.values()]
                        old_xs = [lm.x for lm in last_valid.landmarks.values()]
                        old_ys = [lm.y for lm in last_valid.landmarks.values()]
                        
                        if curr_xs and old_xs:
                            curr_width = max(curr_xs) - min(curr_xs)
                            curr_height = max(curr_ys) - min(curr_ys)
                            old_width = max(old_xs) - min(old_xs)
                            old_height = max(old_ys) - min(old_ys)
                            
                            if old_height > 0 and old_width > 0:
                                curr_area = curr_width * curr_height
                                old_area = old_width * old_height
                                area_ratio = curr_area / old_area
                                
                                if area_ratio > 1.8 or area_ratio < 0.55:
                                    is_valid = False
                                    invalidation_reason = f"Area jump ({area_ratio:.2f}x)"
                                    
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
                    last_valid_skeletons[track_id] = skel
                    if lowpass > 0:
                        last_smoothed = last_smoothed_skeletons.get(track_id)
                        skel = _apply_lowpass(skel, last_smoothed, lowpass)
                    last_smoothed_skeletons[track_id] = skel
                    obj_frames[i].append(skel)
                else:
                    if visualize:
                        print(f"\nSkeleton correction applied at frame {i} track {track_id} ({invalidation_reason})")
                    if last_valid is not None:
                        cached_skel = Skeleton(landmarks=last_valid.landmarks, detection_scale=last_valid.detection_scale, is_cached=True)
                        cached_skel._track_id = track_id
                        obj_frames[i].append(cached_skel)
            else:
                last_valid = last_valid_skeletons.get(track_id)
                if last_valid is not None:
                    cached_skel = Skeleton(landmarks=last_valid.landmarks, detection_scale=last_valid.detection_scale, is_cached=True)
                    cached_skel._track_id = track_id
                    obj_frames[i].append(cached_skel)
                    
        # Add cached skeleton for completely missing tracks
        for track_id, last_valid in last_valid_skeletons.items():
            found_skel = False
            for _, skel in detections:
                if skel is not None and skel._track_id == track_id:
                    found_skel = True
                    break
            if not found_skel and last_valid is not None:
                cached_skel = Skeleton(landmarks=last_valid.landmarks, detection_scale=last_valid.detection_scale, is_cached=True)
                cached_skel._track_id = track_id
                obj_frames[i].append(cached_skel)

        term_key = None
        if has_termios and select.select([sys.stdin], [], [], 0)[0]:
            term_key = sys.stdin.read(1)

        cv_key = -1
        if visualize and i % 3 == 0:
            vis = frame.copy()
            for obj in obj_frames[i]:
                obj.draw(video, vis)
            if not window_created:
                cv2.namedWindow("Extraction & Refining", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Extraction & Refining", 1280, 720)
                window_created = True
            cv2.imshow("Extraction & Refining", vis)
            cv_key = cv2.waitKey(1) & 0xFF

        pressed_v = (term_key == 'v' or cv_key == ord('v'))
        pressed_q = (term_key == 'q' or cv_key == ord('q'))

        if pressed_v:
            visualize = not visualize
            if not visualize and window_created:
                cv2.destroyWindow("Extraction & Refining")
                window_created = False
        elif pressed_q:
            visualize = False
            if window_created:
                cv2.destroyWindow("Extraction & Refining")
                window_created = False



    print()
    if visualize and window_created:
        cv2.destroyWindow("Extraction & Refining")
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
            lambda x: any(word in x.name for word in ball_filter) or getattr(x, 'id', -1) == thrower_id or isinstance(x, Skeleton),
            obj_frame,
        )
        filtered_obj_frames.append(filtered_frame)
    return filtered_obj_frames


def _apply_confidence_filter(skeleton, min_conf):
    """Remove landmarks below the confidence threshold."""
    filtered = {idx: lm for idx, lm in skeleton.landmarks.items() if lm.visibility >= min_conf}
    return Skeleton(landmarks=filtered, detection_scale=skeleton.detection_scale, is_cached=skeleton.is_cached)


def _apply_lowpass(skeleton, prev_skeleton, alpha):
    """Apply first-order IIR low-pass filter (EMA) to keypoint positions.

    smoothed = alpha * previous + (1 - alpha) * current
    alpha=0: no smoothing (raw measurements)
    alpha→1: heavy smoothing (slow to respond)
    """
    if prev_skeleton is None:
        return skeleton

    smoothed_landmarks = {}
    for idx, lm in skeleton.landmarks.items():
        if idx in prev_skeleton.landmarks:
            prev_lm = prev_skeleton.landmarks[idx]
            smoothed_landmarks[idx] = Landmark(
                x=int(alpha * prev_lm.x + (1 - alpha) * lm.x),
                y=int(alpha * prev_lm.y + (1 - alpha) * lm.y),
                visibility=lm.visibility,
            )
        else:
            smoothed_landmarks[idx] = lm

    return Skeleton(landmarks=smoothed_landmarks, detection_scale=skeleton.detection_scale, is_cached=skeleton.is_cached)



def render_video(video: Video, obj_frames, release_frame=-1, release_detector_name="", output_height=720.0):
    orig_height = video.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    if orig_height > 0:
        target_scale = output_height / orig_height
    else:
        target_scale = 1.0
    video.scale = target_scale
    video.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    current_cap_idx = 0
    last_valid_frame = None
    
    last_active_side = "Unknown"
    last_angles = {"ls": None, "le": None, "lk": None, "rs": None, "re": None, "rk": None}
    
    for i, obj_frame in enumerate(obj_frames):
        frame_idx = min(i, len(video) - 1)

        try:
            if frame_idx >= current_cap_idx:
                while current_cap_idx < frame_idx:
                    video.cap.grab()
                    current_cap_idx += 1
                ret, raw_frame = video.cap.read()
                if not ret: raise IndexError
                last_valid_frame = raw_frame if video.scale == 1.0 else cv2.resize(raw_frame, video.size)
                current_cap_idx += 1
            frame = last_valid_frame.copy()
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
    video.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_cap_idx = start_frame
    
    if fps is None:
        fps = video.fps

    last_valid_frame = None
    last_active_side = "Unknown"
    last_angles = {"ls": None, "le": None, "lk": None, "rs": None, "re": None, "rk": None}
    t_start = time.perf_counter()
    
    total_frames = (end_frame - start_frame + 1)
    
    for i in range(total_frames):
        if i % 5 == 0 or i == total_frames - 1:
            elapsed = time.perf_counter() - t_start
            fps_proc = (i + 1) / elapsed if elapsed > 0 else 0
            if fps_proc > 0:
                eta_s = (total_frames - (i + 1)) / fps_proc
                m, s = divmod(int(eta_s), 60)
                eta_str = f" | ETA: {m}m {s:02d}s"
            else:
                eta_str = ""
            print(f"\rRendering frame {i+1}/{total_frames} ({(i+1)/total_frames*100:.1f}%){eta_str}", end="", flush=True)
            
        frame_idx = start_frame + i
            
        if valid_frames is not None and frame_idx not in valid_frames:
            continue
            
        try:
            if frame_idx >= current_cap_idx:
                while current_cap_idx < frame_idx:
                    video.cap.grab()
                    current_cap_idx += 1
                ret, raw_frame = video.cap.read()
                if not ret: raise IndexError
                last_valid_frame = raw_frame if video.scale == 1.0 else cv2.resize(raw_frame, video.size)
                current_cap_idx += 1
            frame = last_valid_frame.copy()
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
                safe_fps = fps if fps and fps > 0 else 30.0
                for prep, rel in cycles:
                    if rel <= frame_idx < rel + int(safe_fps * 1.0):
                        status_text = "RELEASED"
                    elif prep <= frame_idx < min(prep + int(safe_fps * 1.0), rel):
                        status_text = "PREPARE"
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

def export_skeleton_data(obj_frames, output_path, fps, release_frame, start_frame=0, end_frame=None):
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
        
        if end_frame is None:
            end_frame = len(obj_frames) - 1
            
        for i in range(start_frame, end_frame + 1):
            if i >= len(obj_frames):
                break
                
            obj_frame = obj_frames[i]
            
            skel = next((obj for obj in obj_frame if isinstance(obj, Skeleton)), None)
            
            time_sec = (i - start_frame) / fps if fps else 0
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
