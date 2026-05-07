from ultralytics import YOLO

from config import InputParams, YOLOParams, MediaPipeParams
from video import Video
from ml import YOLOFiltered, MediaPipe
from detectors import (
    CombinedThrowerDetector,
    ActionReleaseDetector,
    SkeletonReleaseDetector,
)
from pipeline import (
    extract_obj_frames,
    enrich_player_with_action,
    only_keep_relevant_obj_frames,
    append_thrower_skeleton,
    cut_after_release,
    render_video,
    export_skeleton_data,
    split_video_into_scenes,
)

import argparse
import os
import shutil

def main():
    parser = argparse.ArgumentParser(description="Process basketball videos")
    parser.add_argument("--videos", nargs="+", required=True, help="List of video files or names to process")
    parser.add_argument("--data_path", default="data/", help="Base path for input/output data")
    parser.add_argument("--model_dir", default="models/", help="Directory containing the model files")
    args = parser.parse_args()

    player_filter = ["player", "person", "human"]
    ball_filter = ["ball"]
    yolo_filter = player_filter + ball_filter
    yolo_params = YOLOParams(
        model_path=os.path.join(args.model_dir, "basketball-3-m.pt"), 
        name_filter=yolo_filter
    )
    mp_params = MediaPipeParams(
        model_path=os.path.join(args.model_dir, "pose_landmarker.task"), 
        min_pose_conf=0, 
        min_track_conf=0
    )

    print("Loading models...")
    model = YOLO(yolo_params.model_path)
    yolo_filtered = YOLOFiltered(model, yolo_params.name_filter)
    mediapipe = MediaPipe(mp_params)
    print("Models loaded!\n")

    for video_file in args.videos:
        base_video_id = os.path.splitext(os.path.basename(video_file))[0]
        
        print(f"--- Splitting {base_video_id} into scenes ---")
        input_dir = os.path.join(args.data_path, "input")
        scene_ids = split_video_into_scenes(video_file, input_dir, base_video_id)
        num_scenes = len(scene_ids)
        print(f"Detected {num_scenes} scenes.")
        
        if num_scenes == 1:
            duplicate_path = os.path.join(input_dir, f"{scene_ids[0]}.mp4")
            if os.path.exists(duplicate_path):
                os.remove(duplicate_path)
            scene_ids = [base_video_id]
        else:
            backup_dir = os.path.join(args.data_path, "input-backup")
            os.makedirs(backup_dir, exist_ok=True)
            shutil.move(video_file, os.path.join(backup_dir, os.path.basename(video_file)))
        
        for scene_id in scene_ids:
            print(f"--- Processing Scene: {scene_id} ---")
            params = InputParams(video_id=scene_id, data_path=args.data_path)
            
            video = Video(params.input_video_path, params.output_video_path)

            print(f"Extracting object frames from {params.input_video_path}...")
            obj_frames = extract_obj_frames(video, yolo_filtered)
            print("Extracted!")

            print("Enrich player data with action...")
            obj_frames = enrich_player_with_action(player_filter, obj_frames)
            print("Enriched!")

            print("Find thrower...")
            combined_detector = CombinedThrowerDetector(
                ball_filter=ball_filter, action_filter=["jump-shot"]
            )
            detected_throwers = combined_detector.detect(obj_frames)

            if len(detected_throwers) > 0:
                thrower_id = detected_throwers[0].id
            else:
                thrower_id = -1

            if thrower_id == -1:
                print("No valid thrower detected for this clip! Skipping...")
                video.release()
                continue
            print("Detected!")

            print("Filter object frames...")
            obj_frames = only_keep_relevant_obj_frames(obj_frames, ball_filter, thrower_id)
            print("Filtered!")

            print("Append skeleton of thrower...")
            obj_frames = append_thrower_skeleton(video, obj_frames, thrower_id, mediapipe)
            print("Tracked!")

            print("Cut object frames after release...")
            release_detectors = [ActionReleaseDetector(), SkeletonReleaseDetector()]
            obj_frames, release_frame, release_detector_name = cut_after_release(
                obj_frames, release_detectors, video.fps
            )
            if release_frame == -1:
                print("Release not detected! Skipping...")
                video.release()
                continue
            print("Cut!")

            print(f"Exporting skeleton data to {params.output_data_path}...")
            export_skeleton_data(obj_frames, params.output_data_path, video.fps, release_frame)
            print("Exported!")

            print(f"Writing out to {params.output_video_path}...")
            scene_num = scene_id.split("-")[-1] if num_scenes > 1 else ""
            render_video(video, obj_frames, release_frame, release_detector_name, scene_num)
            print("Wrote!")

            video.release()
            print(f"Finished {scene_id}\n")

    print("All done!")

if __name__ == "__main__":
    main()
