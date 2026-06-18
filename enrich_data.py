import argparse
import os
os.environ["OPENCV_FFMPEG_THREADS"] = "8"
from pathlib import Path

from ml import YOLOPose

# Import the extracted function from isolate_run
from isolate_run import process_video

def main():
    parser = argparse.ArgumentParser(description="Enrich data folder with isolated throws")
    parser.add_argument("data_folder", help="Root data folder containing <experiment>/<participant>/runs/*.mp4")
    parser.add_argument("--model_dir", default="models/", help="Directory containing the model files")
    parser.add_argument("--enable-hud", action="store_true", help="Enable HUD overlays on the output video")
    parser.add_argument("--full-debug-video", action="store_true", help="Always render the full run video with HUD overlays for debugging")
    parser.add_argument("--max-movement", type=float, default=60.0, help="Maximum allowed skeleton movement per frame in scaled pixels")
    parser.add_argument("--max-throws", type=int, default=None, help="Maximum number of throws to detect per video (default: unlimited)")
    parser.add_argument("--pose-model", default="yolov8x-pose-p6.pt", help="YOLO-Pose model filename (in model_dir)")
    parser.add_argument("--enable-invalidation", action="store_true", help="Enable skeleton invalidation logic (defaults to False)")
    parser.add_argument("--output-height", type=float, default=720.0, help="Target height for the output videos")
    parser.add_argument("--visualize", action="store_true", help="Show live OpenCV windows for each processing step (press 'q' to dismiss)")
    parser.add_argument('--min-kp-conf', type=float, default=0.3, help="Minimum keypoint confidence to keep a landmark (0=off, default: 0.3)")
    parser.add_argument('--min-keypoints', type=int, default=6, help="Minimum number of valid landmarks to accept a skeleton (0=off, default: 6)")
    parser.add_argument('--lowpass', type=float, default=0.4, help="Low-pass filter strength on keypoint positions (0=off, 0.8=heavy, default: 0.4)")
    parser.add_argument('--follow-through', type=float, default=0.0, help="Seconds to record after the release (default: 0.0)")
    parser.add_argument('--enable-fallback', action="store_true", help="Enable crop-based fallback tracking for dropped skeletons (slows down processing)")
    args = parser.parse_args()

    data_dir = Path(args.data_folder)
    if not data_dir.exists() or not data_dir.is_dir():
        print(f"Error: Directory '{data_dir}' does not exist.")
        return

    player_filter = ["player", "person", "human"]

    print("Preparing to process videos (model will be loaded per-video to prevent memory leaks)...\n")

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
        
        yolo_pose = YOLOPose(os.path.join(args.model_dir, args.pose_model))
        
        existing_mp4s = list(output_dir.glob(f"{video_path.stem}-*.mp4"))
        existing_csvs = list(output_dir.glob(f"{video_path.stem}-*.csv"))
        
        if existing_mp4s and existing_csvs:
            print(f"Skipping {video_path.name}: Found {len(existing_mp4s)} video(s) and {len(existing_csvs)} CSV(s) already in output directory.")
            continue
        
        process_video(
            str(video_path), 
            str(output_dir), 
            yolo_pose, 
            player_filter,
            enable_hud=args.enable_hud,
            full_debug_video=args.full_debug_video,
            max_movement=args.max_movement,
            output_height=args.output_height,
            enable_invalidation=args.enable_invalidation,
            visualize=args.visualize,
            max_throws=args.max_throws,
            min_kp_conf=args.min_kp_conf,
            min_keypoints=args.min_keypoints,
            lowpass=args.lowpass,
            follow_through=args.follow_through,
            enable_fallback=args.enable_fallback
        )
        
        # Aggressive memory cleanup to prevent FPS drop across videos
        del yolo_pose
        import gc
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("==================================================")
    print("Enrichment complete!")

if __name__ == "__main__":
    main()
