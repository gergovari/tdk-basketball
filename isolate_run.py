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


def main():
    parser = argparse.ArgumentParser(description="Convert a run to throws")
    parser.add_argument("--video", required=True, help="Video file of run to process")
    parser.add_argument(
        "--output_path", default="out/", help="Base path for input/output data"
    )
    parser.add_argument(
        "--model_dir", default="models/", help="Directory containing the model files"
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

    base_video_id = os.path.splitext(os.path.basename(args.video))[0]

    print(f"--- Processing Video: {base_video_id} ---")
    # params = InputParams(video_id=base_video_id, data_path=args.output_path)
    params = {
        "input_video_path": args.video,
        "output_video_path": args.output_path + "/" + base_video_id + ".mp4",
    }

    video = Video(params["input_video_path"], params["output_video_path"])

    print(f"Extracting object frames from {args.video}...")
    obj_frames = extract_obj_frames(video, yolo_filtered)
    print(f"Extracted! ({len(obj_frames)})")

    print("Find thrower...")
    biggest_detector = BiggestPersonThrowerDetector(person_filter=player_filter)
    detected_throwers = biggest_detector.detect(obj_frames)

    if len(detected_throwers) > 0:
        thrower_id = detected_throwers[0].id
    else:
        thrower_id = -1

    if thrower_id == -1:
        print("No valid thrower detected for this clip! Skipping...")
        video.release()
        if os.path.exists(params["output_video_path"]):
            os.remove(params["output_video_path"])
    print("Detected!")

    print("Filter object frames...")
    obj_frames = only_keep_relevant_obj_frames(obj_frames, [], thrower_id)
    print("Filtered!")

    print("Append skeleton of thrower...")
    obj_frames = append_thrower_skeleton(video, obj_frames, thrower_id, mediapipe)
    print("Tracked!")

    print("Detecting prepare-release cycles...")
    prep_detector = SkeletonPrepareDetector()
    rel_detector = SkeletonReleaseDetector()

    cycles = []
    curr_frame = 0
    prepares_found = 0
    releases_found = 0

    while curr_frame < len(obj_frames):
        prep_frame = prep_detector.detect(obj_frames, video.fps, start_idx=curr_frame)
        if prep_frame != -1:
            prepares_found += 1
        else:
            break
            
        rel_frame = rel_detector.detect(obj_frames, video.fps, start_idx=prep_frame)
        if rel_frame != -1:
            releases_found += 1
        else:
            break

        cycles.append((prep_frame, rel_frame))
        curr_frame = rel_frame + 1

    print(f"Found {prepares_found} prepares and {releases_found} releases, resulting in {len(cycles)} full cycles!")

    if not cycles:
        print("No prepare-release cycles detected! Skipping...")
        video.release()
        if os.path.exists(params["output_video_path"]):
            os.remove(params["output_video_path"])
        return

    video.release()
    if os.path.exists(params["output_video_path"]):
        os.remove(params["output_video_path"])

    print("Rendering isolated throws...")
    for cycle_idx, (prep_frame, rel_frame) in enumerate(cycles):
        throw_num = cycle_idx + 1
        out_path = os.path.join(args.output_path, f"{base_video_id}-{throw_num}.mp4")
        print(f"Writing out to {out_path} (frames {prep_frame} to {rel_frame})...")
        render_throw_video(
            params["input_video_path"], 
            out_path, 
            obj_frames, 
            prep_frame, 
            rel_frame, 
            rel_frame, 
            release_detector_name=rel_detector.__class__.__name__, 
            fps=video.fps
        )
        print(f"Wrote throw {throw_num}!")

    print(f"Finished {base_video_id}\n")

    print("All done!")


if __name__ == "__main__":
    main()
