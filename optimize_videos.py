#!/usr/bin/env python3
import sys
import subprocess
from pathlib import Path
import argparse

def optimize_video(input_path, output_path, max_height=None):
    """
    Re-encodes a video for optimal sequential reading in OpenCV.
    
    Why these settings?
    - libx264: Standard, excellent compatibility with OpenCV.
    - preset ultrafast + tune fastdecode: Disables CABAC, in-loop deblocking, and other 
      complex H.264 features. This reduces the CPU load during decoding to an absolute minimum.
    - bf 0: Disables B-frames. B-frames require out-of-order decoding. Since the pipeline
      uses cap.grab() to skip frames, lacking B-frames makes sequential skipping much faster.
    - g 30: Frequent keyframes ensure decoder state can be reset easily.
    - pix_fmt yuv420p: Best color space compatibility.
    """
    cmd = [
        "ffmpeg", "-y", 
        "-hide_banner", "-loglevel", "error", "-stats",
        "-i", str(input_path),
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-tune", "fastdecode",
        "-bf", "0",
        "-g", "30",
        "-crf", "18", # High quality to avoid messing with YOLO/Pose detection
        "-pix_fmt", "yuv420p",
        "-c:a", "copy"
    ]
    
    if max_height:
        # Scale to max_height, keeping aspect ratio. Width must be divisible by 2 for H.264.
        vf_filter = f"scale=-2:'min({max_height},ih)'"
        cmd.extend(["-vf", vf_filter])

    cmd.append(str(output_path))
    
    print(f"Optimizing: {input_path.name} -> {output_path.name}")
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser(
        description="Re-encode videos for maximum OpenCV/pipeline processing speed.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("folder", type=str, help="Folder containing videos to optimize")
    parser.add_argument("--height", type=int, default=None, 
                        help="Optional max height (e.g., 720 or 1080). Downscaling significantly speeds up processing.")
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"Error: '{folder}' is not a valid directory.")
        sys.exit(1)

    video_exts = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
    
    count = 0
    for file in sorted(folder.rglob("*")):
        if file.is_file() and file.suffix.lower() in video_exts and not file.name.startswith(".tmp_opt_"):
            target_file = file.with_suffix(".mp4")
            temp_file = file.with_name(f".tmp_opt_{target_file.name}")
            
            try:
                optimize_video(file, temp_file, args.height)
                
                # If it's a different extension (e.g. .mov -> .mp4), remove the old one
                if file != target_file:
                    file.unlink()
                
                # Replace with the newly optimized mp4 file
                temp_file.replace(target_file)
                count += 1
            except subprocess.CalledProcessError as e:
                print(f"Failed to optimize '{file.name}': {e}")
                if temp_file.exists():
                    temp_file.unlink()
            except KeyboardInterrupt:
                print("\nOptimization cancelled by user.")
                if temp_file.exists():
                    temp_file.unlink()
                sys.exit(1)
                
    print(f"\nDone! Optimized {count} videos in place.")

if __name__ == "__main__":
    main()
