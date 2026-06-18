from ultralytics import YOLO

from config import InputParams, YOLOParams
from video import Video
from ml import YOLOPose
from detectors import (
    BiggestPersonThrowerDetector,
    ThrowCycleDetector,
)
from pipeline import (
    extract_and_refine_obj_frames,
    only_keep_relevant_obj_frames,
    render_throw_video,
    export_skeleton_data,
)

import argparse
import os


def process_video(input_video_path, output_dir, yolo_pose, player_filter, enable_hud=False, full_debug_video=False, max_movement=60.0, output_height=720.0, visualize=False, enable_invalidation=False, max_throws=None, min_kp_conf=0.3, min_keypoints=6, lowpass=0.4, follow_through=0.0, enable_fallback=False):
    if not os.path.isfile(input_video_path):
        print(f"Error: Video file not found: {input_video_path}")
        return

    base_video_id = os.path.splitext(os.path.basename(input_video_path))[0]

    print(f"--- Processing Video: {base_video_id} ---")
    params = {
        "input_video_path": input_video_path,
        "output_video_path": os.path.join(output_dir, base_video_id + ".mp4"),
    }

    video = Video(params["input_video_path"], params["output_video_path"])

    print(f"Extracting object frames from {input_video_path}...")
    yolo_pose.reset()  # Reset tracker state for each new video
    obj_frames = extract_and_refine_obj_frames(video, yolo_pose, 
        max_movement=max_movement, 
        visualize=visualize, 
        enable_invalidation=enable_invalidation, 
        min_kp_conf=min_kp_conf, 
        min_keypoints=min_keypoints, 
        lowpass=lowpass,
        enable_fallback=enable_fallback)
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

    print("Detecting prepare-release cycles...")
    cycle_detector = ThrowCycleDetector(follow_through_seconds=follow_through)
    cycles, prepares_found, releases_found = cycle_detector.detect(
        obj_frames, 
        video.fps, 
        first_thrower_frame, 
        last_thrower_frame, 
        max_throws=max_throws
    )

    print(f"Found {prepares_found} prepares and {releases_found} releases, resulting in {len(cycles)} raw cycles!")

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
        debug_csv_path = os.path.join(output_dir, f"{base_video_id}-DEBUG.csv")
        export_skeleton_data(
            obj_frames,
            debug_csv_path,
            video.fps,
            release_frame=-1,
            start_frame=0,
            end_frame=len(obj_frames) - 1
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
        csv_path = os.path.join(output_dir, f"{base_video_id}-{identifier}.csv")
        export_skeleton_data(
            obj_frames, 
            csv_path, 
            video.fps, 
            release_frame=-1, 
            start_frame=first_thrower_frame, 
            end_frame=last_thrower_frame
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
        csv_path = os.path.join(output_dir, f"{base_video_id}-{throw_num}.csv")
        export_skeleton_data(
            obj_frames, 
            csv_path, 
            video.fps, 
            release_frame=rel_frame, 
            start_frame=prep_frame, 
            end_frame=rel_frame
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
        "--full-debug-video", action="store_true", help="Always render the full run video with HUD overlays for debugging"
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Show live OpenCV windows for each processing step (press 'q' to dismiss)"
    )
    parser.add_argument(
        "--enable-invalidation", action="store_true", help="Enable skeleton invalidation logic (defaults to False)"
    )
    parser.add_argument(
        '--max-movement', type=float, default=60.0, help="Maximum allowed skeleton movement per frame in scaled pixels"
    )
    parser.add_argument(
        '--max-throws', type=int, default=None, help="Maximum number of throws to detect per video (default: unlimited)"
    )
    parser.add_argument(
        '--pose-model', default="yolov8x-pose-p6.pt", help="YOLO-Pose model filename (in model_dir)"
    )
    parser.add_argument(
        "--output-height", type=float, default=720.0, help="Target height for the output videos"
    )
    parser.add_argument(
        '--min-kp-conf', type=float, default=0.3, help="Minimum keypoint confidence to keep a landmark (0=off, default: 0.3)"
    )
    parser.add_argument(
        '--min-keypoints', type=int, default=6, help="Minimum number of valid landmarks to accept a skeleton (0=off, default: 6)"
    )
    parser.add_argument(
        '--lowpass', type=float, default=0.4, help="Low-pass filter strength on keypoint positions (0=off, 0.8=heavy, default: 0.4)"
    )
    parser.add_argument(
        '--follow-through', type=float, default=0.0, help="Seconds to record after the release (default: 0.0)"
    )
    parser.add_argument(
        '--enable-fallback', action="store_true", help="Enable crop-based fallback tracking for dropped skeletons (slows down processing)"
    )
    args = parser.parse_args()

    player_filter = ["player", "person", "human"]

    print("Loading model...")
    yolo_pose = YOLOPose(os.path.join(args.model_dir, args.pose_model))
    print(f"Model loaded! (device: {yolo_pose.device})\n")

    process_video(args.video, args.output_path, yolo_pose, player_filter, enable_hud=args.enable_hud, full_debug_video=args.full_debug_video, max_movement=args.max_movement, output_height=args.output_height, visualize=args.visualize, enable_invalidation=args.enable_invalidation, max_throws=args.max_throws, min_kp_conf=args.min_kp_conf, min_keypoints=args.min_keypoints, lowpass=args.lowpass, follow_through=args.follow_through, enable_fallback=args.enable_fallback)
    print("All done!")


if __name__ == "__main__":
    main()
