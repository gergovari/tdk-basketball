import argparse
import os
from pathlib import Path

from ultralytics import YOLO
from config import YOLOParams, MediaPipeParams
from ml import YOLOFiltered, MediaPipe, YOLOPose

# Import the extracted function from isolate_run
from isolate_run import process_video

def main():
    parser = argparse.ArgumentParser(description="Enrich data folder with isolated throws")
    parser.add_argument("data_folder", help="Root data folder containing <experiment>/<participant>/runs/*.mp4")
    parser.add_argument("--model_dir", default="models/", help="Directory containing the model files")
    parser.add_argument("--enable-hud", action="store_true", help="Enable HUD overlays on the output video")
    parser.add_argument("--always-split", action="store_true", help="Always output individual throw videos, ignoring the exactly-5 rule")
    parser.add_argument("--full-debug-video", action="store_true", help="Always render the full run video with HUD overlays for debugging")
    parser.add_argument("--max-movement", type=float, default=60.0, help="Maximum allowed skeleton movement per frame in scaled pixels")
    parser.add_argument("--pose-backend", choices=["mediapipe", "yolo"], default="yolo", help="Pose estimation backend: 'mediapipe' (CPU) or 'yolo' (GPU, faster)")
    parser.add_argument("--enable-invalidation", action="store_true", help="Enable skeleton invalidation logic (defaults to False)")
    parser.add_argument("--output-height", type=float, default=720.0, help="Target height for the output videos")
    parser.add_argument("--visualize", action="store_true", help="Show live OpenCV windows for each processing step (press 'q' to dismiss)")
    args = parser.parse_args()

    data_dir = Path(args.data_folder)
    if not data_dir.exists() or not data_dir.is_dir():
        print(f"Error: Directory '{data_dir}' does not exist.")
        return

    player_filter = ["player", "person", "human"]
    yolo_filter = player_filter
    yolo_params = YOLOParams(
        model_path=os.path.join(args.model_dir, "yolo26n.pt"),
        name_filter=yolo_filter,
    )

    print("Loading models once for all videos...")
    model = YOLO(yolo_params.model_path)
    yolo_filtered = YOLOFiltered(model, yolo_params.name_filter)
    if args.pose_backend == "yolo":
        pose_model = YOLOPose(os.path.join(args.model_dir, "yolo11n-pose.pt"))
        print(f"Pose backend: YOLO-Pose (GPU)")
    else:
        mp_params = MediaPipeParams(
            model_path=os.path.join(args.model_dir, "pose_landmarker.task"),
            min_pose_conf=0,
            min_track_conf=0,
        )
        pose_model = MediaPipe(mp_params)
        print(f"Pose backend: MediaPipe (CPU)")
    print("Models loaded!\n")

    # Collect all video files
    video_files = []
    for ext in ["*.mp4", "*.MP4", "*.mov", "*.MOV"]:
        # Find paths matching <data_dir>/<experiment>/<participant>/runs/<video>
        # The glob pattern "*/*/runs/<ext>" ensures it matches the directory structure exactly
        for p in data_dir.glob(f"*/*/runs/{ext}"):
            video_files.append(p)

    if not video_files:
        print(f"No videos found matching the structure in {data_dir}/*/*/runs/")
        return

    total_videos = len(video_files)
    import time
    batch_t_start = time.perf_counter()
    for video_idx, video_path in enumerate(video_files):
        # e.g., <data_dir>/<exp_id>/<part_id>/runs/<video_file>
        runs_dir = video_path.parent
        participant_dir = runs_dir.parent
        
        output_dir = participant_dir / "throw"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("==================================================")
        elapsed = time.perf_counter() - batch_t_start
        if video_idx > 0:
            eta_s = (elapsed / video_idx) * (total_videos - video_idx)
            m, s = divmod(int(eta_s), 60)
            h, m = divmod(m, 60)
            eta_str = f" | ETA: {h}h {m}m {s:02d}s" if h > 0 else f" | ETA: {m}m {s:02d}s"
        else:
            eta_str = ""
        print(f"[Video {video_idx + 1}/{total_videos}] ({(video_idx + 1) / total_videos * 100:.1f}%){eta_str}")
        print(f"Processing video: {video_path}")
        print(f"Output directory: {output_dir}")
        
        process_video(
            str(video_path), 
            str(output_dir), 
            yolo_filtered, 
            pose_model, 
            player_filter,
            enable_hud=args.enable_hud,
            always_split=args.always_split,
            full_debug_video=args.full_debug_video,
            max_movement=args.max_movement,
            output_height=args.output_height,
            enable_invalidation=args.enable_invalidation,
            visualize=args.visualize
        )

    print("==================================================")
    print("Enrichment complete!")

if __name__ == "__main__":
    main()
