import cv2
import os
import argparse

def split_video_into_scenes(input_path, output_dir, base_id, threshold=0.85):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    scene_idx = 0
    prev_hist = None
    frames_in_current_scene = 0
    min_scene_frames = int(fps)
    valid_scenes = []
    
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{base_id}-{scene_idx}.mp4")
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frames_in_current_scene += 1
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

        if prev_hist is not None:
            similarity = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            if similarity < threshold and frames_in_current_scene > min_scene_frames:
                out.release()
                valid_scenes.append(scene_idx)
                scene_idx += 1
                out_path = os.path.join(output_dir, f"{base_id}-{scene_idx}.mp4")
                out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
                frames_in_current_scene = 0
                
        out.write(frame)
        prev_hist = hist

    out.release()
    cap.release()
    
    if frames_in_current_scene > 0:
        if frames_in_current_scene < min_scene_frames and scene_idx > 0:
            os.remove(os.path.join(output_dir, f"{base_id}-{scene_idx}.mp4"))
        else:
            valid_scenes.append(scene_idx)
            
    return [f"{base_id}-{i}" for i in valid_scenes]

def main():
    parser = argparse.ArgumentParser(description="Split a video into scenes")
    parser.add_argument("--video", required=True, help="Input video file")
    parser.add_argument("--output_dir", required=True, help="Output directory for scenes")
    parser.add_argument("--threshold", type=float, default=0.85, help="Threshold for scene change detection")
    args = parser.parse_args()

    base_id = os.path.splitext(os.path.basename(args.video))[0]
    print(f"Splitting {args.video} into scenes...")
    scenes = split_video_into_scenes(args.video, args.output_dir, base_id, args.threshold)
    print(f"Detected {len(scenes)} scenes:")
    for scene in scenes:
        print(f" - {scene}")

if __name__ == "__main__":
    main()
