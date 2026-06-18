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


def extract_obj_frames(video: Video, yolo_pose: YOLOPose, visualize=False):
    """Run YOLOPose on the video to detect people AND extract their skeletons in a single pass.

    Returns obj_frames where each frame contains Object and Skeleton entries.
    Each detected person contributes an Object (bounding box) and optionally
    a Skeleton (keypoints) — both produced by the same model call.
    """
    total_frames = len(video)
    obj_frames = [[] for _ in range(total_frames)]
    # Run YOLO-Pose at ~15 effective fps — plenty for tracking people standing in place
    yolo_stride = max(1, round(video.fps / 15))
    t_start = time.perf_counter()
    last_detections = []
    window_created = False
    video.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
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
            print(f"\rExtracting frame {i+1}/{total_frames} ({(i+1)/total_frames*100:.1f}%) — {fps:.1f} fps (video: {video.fps} fps, stride: {yolo_stride}){eta_str}", end="", flush=True)
        if i % yolo_stride == 0:
            # Full decode + YOLO-Pose inference
            ret, frame = video.cap.read()
            if not ret:
                break
            detections = yolo_pose.track(video, frame)
            # Flatten (Object, Skeleton) pairs into a single list
            last_detections = []
            for obj, skel in detections:
                last_detections.append(obj)
                if skel is not None:
                    # Tag the skeleton with the same track ID so we can
                    # associate it with its Object later
                    skel._track_id = obj.id
                    last_detections.append(skel)
        else:
            # Fast skip — advances decoder without producing a numpy array
            video.cap.grab()
        obj_frames[i].extend(last_detections)
        if visualize and i % yolo_stride == 0:
            vis = frame.copy()
            for obj in obj_frames[i]:
                obj.draw(video, vis)
            if not window_created:
                cv2.namedWindow("YOLO Detection", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("YOLO Detection", 1280, 720)
                window_created = True
            cv2.imshow("YOLO Detection", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                visualize = False
                cv2.destroyWindow("YOLO Detection")
                window_created = False
    print()
    if visualize and window_created:
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


def refine_thrower_skeleton(video: Video, obj_frames, thrower_id, yolo_pose: YOLOPose,
                            max_movement=60.0, visualize=False, enable_invalidation=False,
                            min_kp_conf=0.3, min_keypoints=6, lowpass=0.4):
    """Ensure every frame has a skeleton for the thrower.

    The initial extract_obj_frames pass already produced skeletons alongside
    bounding boxes. This function:
    1. Keeps the skeleton that belongs to the thrower (matching by track ID).
    2. For frames where the thrower's skeleton is missing (tracking dropout),
       runs YOLOPose on a crop of the last known bounding box as fallback.
    3. Applies configurable filters:
       - Confidence filter: drops keypoints below min_kp_conf
       - Minimum keypoints: rejects skeletons with fewer than min_keypoints valid landmarks
       - Invalidation: rejects skeletons with implausible jumps (area/movement)
       - Low-pass filter: first-order IIR (EMA) on keypoint positions to reduce jitter
    4. Caches the last valid skeleton for complete dropouts.
    """
    last_valid_skeleton = None  # Last skeleton that passed all filters (pre-lowpass)
    last_smoothed_skeleton = None  # Last skeleton after lowpass (used as lowpass state)
    last_known_rect = None
    total_frames = len(video)
    fps = video.fps or 30
    t_start = time.perf_counter()
    window_created = False

    filter_desc = []
    if min_kp_conf > 0:
        filter_desc.append(f"kp_conf≥{min_kp_conf}")
    if min_keypoints > 0:
        filter_desc.append(f"min_kp≥{min_keypoints}")
    if enable_invalidation:
        filter_desc.append(f"invalidation(max_move={max_movement})")
    if lowpass > 0:
        filter_desc.append(f"lowpass(α={lowpass})")
    print(f"  Filters: {', '.join(filter_desc) if filter_desc else 'none'}")

    # Calculate the "usual spot" by finding the median center of the thrower
    thrower_centers = []
    for frame_objs in obj_frames:
        for obj in frame_objs:
            if hasattr(obj, 'id') and obj.id == thrower_id:
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

    # Determine upfront which frames need crop fallback (no thrower skeleton from extraction)
    # so we know which frames require actual pixel decoding.
    needs_fallback = set()
    for i in range(total_frames):
        has_thrower_skel = any(
            isinstance(obj, Skeleton) and getattr(obj, '_track_id', -1) == thrower_id
            for obj in obj_frames[i]
        )
        if not has_thrower_skel:
            needs_fallback.add(i)

    video.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

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
            print(f"\rRefining skeleton {i+1}/{total_frames} ({(i+1)/total_frames*100:.1f}%){eta_str}", end="", flush=True)
        show_this_frame = visualize

        # Only decode a frame if we need pixels (fallback inference or visualization)
        need_pixels = (i in needs_fallback) or show_this_frame
        frame = None
        if need_pixels:
            video.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = video.cap.read()
            if not ret:
                frame = None

        current_thrower = next(
            (obj for obj in obj_frames[i] if hasattr(obj, 'id') and not isinstance(obj, Skeleton) and obj.id == thrower_id), None
        )

        # Stop tracking if they move too far from the usual spot
        if current_thrower is not None and usual_spot is not None:
            cx = (current_thrower.rect.x1 + current_thrower.rect.x2) / 2
            cy = (current_thrower.rect.y1 + current_thrower.rect.y2) / 2
            dist_from_usual = math.hypot(cx - usual_spot[0], cy - usual_spot[1])
            if dist_from_usual > video.size[0] * 0.15:
                current_thrower = None
                last_valid_skeleton = None
                last_smoothed_skeleton = None

        # Check if we already have a skeleton for the thrower from the extraction pass
        existing_skel = next(
            (obj for obj in obj_frames[i] if isinstance(obj, Skeleton) and getattr(obj, '_track_id', -1) == thrower_id),
            None
        )

        # Remove ALL skeletons from this frame — we'll add back only the thrower's
        obj_frames[i] = [obj for obj in obj_frames[i] if not isinstance(obj, Skeleton)]

        thrower_skeleton = existing_skel

        # If no skeleton from extraction, try crop-based fallback (requires pixels)
        if thrower_skeleton is None and frame is not None:
            if current_thrower is not None:
                rect = current_thrower.rect.with_padding(
                    ScaledInt(60, (1280, 720), video.size).value
                )
                last_known_rect = rect
            elif last_known_rect is not None:
                rect = last_known_rect
            else:
                rect = None

            if rect is not None:
                thrower_skeleton = yolo_pose.detect_on_crop(frame, rect, video.scale)
        elif thrower_skeleton is not None:
            # Update last_known_rect from the thrower's bounding box
            if current_thrower is not None:
                last_known_rect = current_thrower.rect.with_padding(
                    ScaledInt(60, (1280, 720), video.size).value
                )

        if thrower_skeleton is not None:
            # --- Filter 1: Confidence threshold ---
            if min_kp_conf > 0:
                thrower_skeleton = _apply_confidence_filter(thrower_skeleton, min_kp_conf)

            # --- Filter 2: Minimum keypoints ---
            if min_keypoints > 0 and len(thrower_skeleton.landmarks) < min_keypoints:
                thrower_skeleton = None  # Reject — will fall through to cache

        if thrower_skeleton is not None:
            # --- Filter 3: Invalidation (area/movement jumps) ---
            is_valid = True
            invalidation_reason = ""
            if enable_invalidation and last_valid_skeleton is not None:
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

                            # Scale-invariant Area Check (Resolution independent)
                            if area_ratio > 1.8 or area_ratio < 0.55:
                                is_valid = False
                                invalidation_reason = f"Area jump ({area_ratio:.2f}x)"

                            # Dynamic Movement Threshold (scaled to the thrower's true pixel height)
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
                last_valid_skeleton = thrower_skeleton

                # --- Filter 4: Low-pass filter ---
                if lowpass > 0:
                    thrower_skeleton = _apply_lowpass(thrower_skeleton, last_smoothed_skeleton, lowpass)

                last_smoothed_skeleton = thrower_skeleton
                obj_frames[i].append(thrower_skeleton)
            else:
                print(f"\nSkeleton correction applied at frame {i} ({invalidation_reason})")
                cached_skel = Skeleton(landmarks=last_valid_skeleton.landmarks, detection_scale=last_valid_skeleton.detection_scale, is_cached=True)
                obj_frames[i].append(cached_skel)
        elif last_valid_skeleton is not None:
            cached_skel = Skeleton(landmarks=last_valid_skeleton.landmarks, detection_scale=last_valid_skeleton.detection_scale, is_cached=True)
            obj_frames[i].append(cached_skel)

        if show_this_frame and frame is not None and i % 3 == 0:
            vis = frame.copy()
            for obj in obj_frames[i]:
                obj.draw(video, vis)
            if not window_created:
                cv2.namedWindow("Skeleton Tracking", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Skeleton Tracking", 1280, 720)
                window_created = True
            cv2.imshow("Skeleton Tracking", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                visualize = False
                cv2.destroyWindow("Skeleton Tracking")
                window_created = False

    print()  # Newline after progress finishes
    if visualize and window_created:
        cv2.destroyWindow("Skeleton Tracking")
    return obj_frames

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
                for prep, rel in cycles:
                    if prep <= frame_idx < rel:
                        status_text = "PREPARE"
                    elif frame_idx >= rel:
                        status_text = "RELEASED"
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
