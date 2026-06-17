from ultralytics import YOLO

from config import InputParams, YOLOParams, MediaPipeParams
from video import Video
from ml import YOLOFiltered, MediaPipe
from detectors import (
    BiggestPersonThrowerDetector,
    SkeletonReleaseDetector,
    SkeletonPrepareDetector,
)
from pipeline import (
    extract_obj_frames,
    only_keep_relevant_obj_frames,
    append_thrower_skeleton,
    render_throw_video,
    # export_skeleton_data,
)

import argparse
import os


def process_video(input_video_path, output_dir, yolo_filtered, mediapipe, player_filter, enable_hud=False, always_split=False, full_debug_video=False, max_movement=25.0, output_height=720.0):
    base_video_id = os.path.splitext(os.path.basename(input_video_path))[0]

    print(f"--- Processing Video: {base_video_id} ---")
    params = {
        "input_video_path": input_video_path,
        "output_video_path": os.path.join(output_dir, base_video_id + ".mp4"),
    }

    video = Video(params["input_video_path"], params["output_video_path"])

    print(f"Extracting object frames from {input_video_path}...")
    obj_frames = extract_obj_frames(video, yolo_filtered)
    print(f"Extracted! ({len(obj_frames)})")

    print("Find thrower...")
    biggest_detector = BiggestPersonThrowerDetector(person_filter=player_filter)
    detected_throwers = biggest_detector.detect(obj_frames, video.size)

    if len(detected_throwers) > 0:
        thrower_id = detected_throwers[0].id
    else:
        thrower_id = -1

    if thrower_id == -1:
        print("No valid thrower detected for this clip! Aborting analysis and rendering as NO...")
        
        if full_debug_video:
            print(f"Rendering full raw debug video with HUD...")
            debug_out_path = os.path.join(output_dir, f"{base_video_id}-DEBUG.mp4")
            render_throw_video(
                params["input_video_path"], 
                debug_out_path, 
                obj_frames, 
                0, 
                len(obj_frames) - 1, 
                -1, 
                fps=video.fps,
                enable_hud=True,
                output_height=output_height
            )
            print(f"Wrote debug video to {debug_out_path}!")
            
        video.release()
        if os.path.exists(params["output_video_path"]):
            os.remove(params["output_video_path"])
            
        out_path = os.path.join(output_dir, f"{base_video_id}-NO.mp4")
        render_throw_video(
            params["input_video_path"], 
            out_path, 
            obj_frames, 
            0, 
            len(obj_frames) - 1, 
            -1, 
            fps=video.fps,
            enable_hud=enable_hud,
            output_height=output_height
        )
        print(f"Finished {base_video_id}\n")
        return
    print("Detected!")

    # Find frames where the thrower is actively visible, bridging small tracking dropouts
    raw_valid_frames = [i for i, frame in enumerate(obj_frames) if any(getattr(obj, "id", -1) == thrower_id for obj in frame)]
    valid_frames_set = set(raw_valid_frames)
    max_gap = int(video.fps)
    
    for i in range(len(raw_valid_frames) - 1):
        gap = raw_valid_frames[i+1] - raw_valid_frames[i]
        if 1 < gap <= max_gap:
            for j in range(raw_valid_frames[i] + 1, raw_valid_frames[i+1]):
                valid_frames_set.add(j)
                
    first_thrower_frame = raw_valid_frames[0]
    last_thrower_frame = raw_valid_frames[-1]

    print("Filter object frames...")
    obj_frames = only_keep_relevant_obj_frames(obj_frames, [], thrower_id)
    print("Filtered!")

    print("Append skeleton of thrower...")
    obj_frames = append_thrower_skeleton(video, obj_frames, thrower_id, mediapipe, max_movement)
    print("Tracked!")

    print("Detecting prepare-release cycles...")
    prep_detector = SkeletonPrepareDetector()
    rel_detector = SkeletonReleaseDetector()

    cycles = []
    curr_frame = first_thrower_frame
    prepares_found = 0
    releases_found = 0

    while curr_frame <= last_thrower_frame:
        # 1. Search forward to find any prepare phase (this naturally skips the previous shot's follow-through)
        prep_forward = prep_detector.detect(obj_frames, video.fps, start_idx=curr_frame)
        if prep_forward == -1:
            break
            
        # 2. Find the release that follows
        rel_frame = rel_detector.detect(obj_frames, video.fps, start_idx=prep_forward)
        if rel_frame != -1:
            releases_found += 1
        else:
            break
            
        # 3. Search backward from the release to find the exact peak of the dip
        prep_frame = prep_detector.detect_backward(obj_frames, video.fps, start_idx=prep_forward, end_idx=rel_frame)
        if prep_frame != -1:
            # Add a 0.5 second lead-up to show the full downward motion before the peak dip
            prep_frame = max(0, prep_frame - int(video.fps * 0.5))
            prepares_found += 1
        else:
            # Fallback in case a perfect prepare phase isn't found
            prep_frame = max(prep_forward, rel_frame - int(1.5 * video.fps))
            prepares_found += 1

        cycles.append((prep_frame, rel_frame))
        curr_frame = rel_frame + 1

    print(f"Found {prepares_found} prepares and {releases_found} releases, resulting in {len(cycles)} raw cycles!")

    # Merge overlapping throws (e.g. pump fakes or double clutches)
    merged_cycles = []
    for cycle in cycles:
        if not merged_cycles:
            merged_cycles.append(cycle)
        else:
            last_prep, last_rel = merged_cycles[-1]
            curr_prep, curr_rel = cycle
            
            # If the next throw starts before or very shortly after the previous release (e.g. within 1 second)
            if curr_prep <= last_rel + int(video.fps * 1.0):
                merged_cycles[-1] = (last_prep, curr_rel)
            else:
                merged_cycles.append(cycle)
                
    cycles = merged_cycles
    print(f"After merging overlapping actions, we have {len(cycles)} definitive throw cycles.")

    video.release()
    if os.path.exists(params["output_video_path"]):
        os.remove(params["output_video_path"])

    if full_debug_video:
        print(f"Rendering full raw debug video with HUD...")
        debug_out_path = os.path.join(output_dir, f"{base_video_id}-DEBUG.mp4")
        render_throw_video(
            params["input_video_path"], 
            debug_out_path, 
            obj_frames, 
            0, 
            len(obj_frames) - 1, 
            -1, 
            fps=video.fps,
            enable_hud=True,
            cycles=cycles,
            output_height=output_height
        )
        print(f"Wrote debug video to {debug_out_path}!")

    if len(cycles) == 0:
        identifier = "NO"
        print(f"Detected 0 throws. Rendering full video in one piece as {identifier}...")
        out_path = os.path.join(output_dir, f"{base_video_id}-{identifier}.mp4")
        
        render_throw_video(
            params["input_video_path"], 
            out_path, 
            obj_frames, 
            first_thrower_frame, 
            last_thrower_frame, 
            -1, 
            fps=video.fps,
            enable_hud=enable_hud,
            valid_frames=valid_frames_set,
            output_height=output_height
        )
        print(f"Finished {base_video_id}\n")
        return
        
    if not always_split and len(cycles) != 5:
        identifier = "MORE" if len(cycles) > 5 else "PARTIAL"
        print(f"Detected {len(cycles)} throws. Rendering full video in one piece as {identifier}...")
        out_path = os.path.join(output_dir, f"{base_video_id}-{identifier}.mp4")
        
        render_throw_video(
            params["input_video_path"], 
            out_path, 
            obj_frames, 
            first_thrower_frame, 
            last_thrower_frame, 
            -1, 
            fps=video.fps,
            enable_hud=enable_hud,
            valid_frames=valid_frames_set,
            output_height=output_height
        )
        print(f"Finished {base_video_id}\n")
        return

    print("Rendering isolated throws...")
    for cycle_idx, (prep_frame, rel_frame) in enumerate(cycles):
        throw_num = cycle_idx + 1
        out_path = os.path.join(output_dir, f"{base_video_id}-{throw_num}.mp4")
        print(f"Writing out to {out_path} (frames {prep_frame} to {rel_frame})...")
        render_throw_video(
            input_video_path,
            out_path,
            obj_frames,
            start_frame=prep_frame,
            end_frame=rel_frame,
            release_frame=rel_frame,
            fps=video.fps,
            enable_hud=enable_hud,
            valid_frames=valid_frames_set,
            output_height=output_height
        )
        print(f"Wrote throw {throw_num}!")

    print(f"Finished {base_video_id}\n")

def main():
    parser = argparse.ArgumentParser(description="Convert a run to throws")
    parser.add_argument("--video", required=True, help="Video file of run to process")
    parser.add_argument(
        "--output_path", default="out/", help="Base path for input/output data"
    )
    parser.add_argument(
        "--model_dir", default="models/", help="Directory containing the model files"
    )
    parser.add_argument(
        "--enable-hud", action="store_true", help="Enable HUD overlays on the output video"
    )
    parser.add_argument(
        "--always-split", action="store_true", help="Always output individual throw videos, ignoring the exactly-5 rule"
    )
    parser.add_argument(
        "--full-debug-video", action="store_true", help="Always render the full run video with HUD overlays for debugging"
    )
    parser.add_argument(
        "--max-movement", type=float, default=40.0, help="Maximum allowed skeleton movement per frame in scaled pixels"
    )
    parser.add_argument(
        "--output-height", type=float, default=720.0, help="Target height for the output videos"
    )
    args = parser.parse_args()

    player_filter = ["player", "person", "human"]
    yolo_filter = player_filter
    yolo_params = YOLOParams(
        model_path=os.path.join(args.model_dir, "yolo26n.pt"),
        name_filter=yolo_filter,
    )
    mp_params = MediaPipeParams(
        model_path=os.path.join(args.model_dir, "pose_landmarker.task"),
        min_pose_conf=0,
        min_track_conf=0,
    )

    print("Loading models...")
    model = YOLO(yolo_params.model_path)
    yolo_filtered = YOLOFiltered(model, yolo_params.name_filter)
    mediapipe = MediaPipe(mp_params)
    print("Models loaded!\n")

    process_video(args.video, args.output_path, yolo_filtered, mediapipe, player_filter, enable_hud=args.enable_hud, always_split=args.always_split, full_debug_video=args.full_debug_video, max_movement=args.max_movement, output_height=args.output_height)
    print("All done!")


if __name__ == "__main__":
    main()
