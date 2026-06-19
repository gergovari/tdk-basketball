import argparse
import os
import glob
import csv
import cv2
import math
import time
import gc
import multiprocessing as mp_lib
from ui import HUD
from entities import Skeleton, Landmark as EntityLandmark

class MockVideo:
    scale = 1.0

def process_video_worker(args_tuple):
    import os
    # Silence C++ GLOG spam from MediaPipe workers
    try:
        fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(fd, 2)
    except Exception:
        pass
        
    if len(args_tuple) == 5:
        video_path, output_path, landmark_mapping, enable_hud, log_queue = args_tuple
    else:
        video_path, output_path, landmark_mapping, enable_hud = args_tuple
        log_queue = None
        
    return process_video(video_path, output_path, landmark_mapping, enable_hud, is_parallel=True, log_queue=log_queue)

def process_video(video_path, output_path, landmark_mapping, enable_hud=False, is_parallel=False, log_queue=None):
    if not os.path.isfile(video_path):
        return f"Error: Video file not found: {video_path}"

    base_video_id = os.path.splitext(os.path.basename(video_path))[0]
    
    # Find all CSVs for this video
    search_pattern = os.path.join(output_path, f"{base_video_id}-*.csv")
    existing_csvs = glob.glob(search_pattern)
    existing_csvs = [f for f in existing_csvs if "-mp-" not in f]
    
    if not existing_csvs:
        return f"Skipped {base_video_id} (No throws found or already processed)"

    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    
    try:
        from tqdm import tqdm
        def safe_print(msg):
            if log_queue is not None:
                log_queue.put(msg)
            else:
                tqdm.write(msg)
    except ImportError:
        def safe_print(msg):
            if log_queue is not None:
                log_queue.put(msg)
            else:
                print(msg)
    
    model_path = os.path.join(os.path.dirname(__file__), "pose_landmarker_heavy.task")
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False,
        num_poses=5, # Detect multiple people, we will filter for the thrower
        running_mode=vision.RunningMode.VIDEO
    )

    if not is_parallel:
        print(f"\n==================================================")
        print(f"Processing video: {video_path}")
        print(f"Output directory: {output_path}")
        print(f"Found {len(existing_csvs)} throws to process for {base_video_id}...")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    t_process_start = time.perf_counter()
    throws_processed = 0

    yolo_indices = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

    for csv_path in sorted(existing_csvs):
        basename = os.path.basename(csv_path)
        name_no_ext = os.path.splitext(basename)[0]
        
        # Remove trailing -m or -h if present to get the unique name
        original_suffix = ""
        if name_no_ext.endswith('-m') or name_no_ext.endswith('-h'):
            original_suffix = name_no_ext[-2:]
            unique_name = name_no_ext[:-2]
        else:
            unique_name = name_no_ext
            
        if not is_parallel:
            print(f"Processing throw {unique_name}...")
        else:
            safe_print(f"[{base_video_id}] Starting throw {unique_name}...")
            
        out_csv_path = os.path.join(output_path, f"{unique_name}-mp{original_suffix}.csv")
        out_video_path = os.path.join(output_path, f"{unique_name}-mp{original_suffix}.mp4")
        
        video_writer = None
        if enable_hud:
            video_writer = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        
        frame_data = []
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            
            frame_idx = header.index("frame")
            time_idx = header.index("time_sec")
            rel_idx = header.index("released")
            hand_idx = header.index("handedness")
            
            for row in reader:
                f_d = {
                    'frame': int(row[frame_idx]),
                    'time_sec': row[time_idx],
                    'released': row[rel_idx],
                    'handedness': row[hand_idx],
                    'yolo': {}
                }
                # Load YOLO normalized coords for matching
                for y_idx in yolo_indices:
                    name = landmark_mapping.get(y_idx)
                    if name and f"{name}_x" in header and f"{name}_y" in header:
                        try:
                            f_d['yolo'][y_idx] = {
                                'x': float(row[header.index(f"{name}_x")]),
                                'y': float(row[header.index(f"{name}_y")])
                            }
                        except (ValueError, IndexError):
                            pass
                frame_data.append(f_d)

        if not frame_data:
            if not is_parallel:
                print(f"Warning: {csv_path} is empty, skipping.")
            continue

        start_frame = frame_data[0]['frame']
        
        with open(out_csv_path, 'w', newline='') as out_f:
            writer = csv.writer(out_f)
            headers = ["frame", "time_sec", "released", "handedness"]
            for idx in sorted(landmark_mapping.keys()):
                name = landmark_mapping[idx]
                headers.extend([f"{name}_x", f"{name}_y", f"{name}_conf"])
            writer.writerow(headers)

            last_coords = {idx: ["", "", ""] for idx in landmark_mapping.keys()}
            
            with vision.PoseLandmarker.create_from_options(options) as detector:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                current_cap_idx = start_frame
            
                t_start = time.perf_counter()
                total_frames = len(frame_data)
                
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
                            
                        if not is_parallel:
                            print(f"\rExtracting MediaPipe {i+1}/{total_frames} ({(i+1)/total_frames*100:.1f}%) - {fps_proc:.1f} fps{eta_str}", end="", flush=True)
                        elif i % 15 == 0 or i == total_frames - 1:
                            safe_print(f"[{base_video_id}] {unique_name}: {i+1}/{total_frames} ({(i+1)/total_frames*100:.1f}%) - {fps_proc:.1f} fps")

                    f_data = frame_data[i]
                    current_frame_num = f_data['frame']
                    
                    frame_to_process = None
                    while current_cap_idx <= current_frame_num:
                        ret, f = cap.read()
                        if not ret: break
                        frame_to_process = f
                        current_cap_idx += 1
                        
                    if frame_to_process is None:
                        if not is_parallel:
                            print(f"Warning: Could not read frame {current_frame_num}")
                        break
                        
                    frame = frame_to_process
                    
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
                    timestamp_ms = int(current_frame_num * (1000.0 / fps))
                    results = detector.detect_for_video(mp_image, timestamp_ms)
                    
                    row = [current_frame_num, f_data['time_sec'], f_data['released'], f_data['handedness']]
                    
                    best_lms = None
                    best_extracted_landmarks = None
                    
                    if results.pose_landmarks:
                        frame_h, frame_w, _ = frame.shape
                        min_mse = float('inf')
                        
                        # Find the pose that best matches the YOLO thrower
                        for pose_idx, lms in enumerate(results.pose_landmarks):
                            extracted = {}
                            for idx in landmark_mapping.keys():
                                if idx < len(lms):
                                    lm = lms[idx]
                                    extracted[idx] = {
                                        'x': lm.x * frame_w,
                                        'y': lm.y * frame_h,
                                        'conf': lm.visibility
                                    }
                            
                            # Normalize this pose candidate to compare with YOLO
                            l_hip = extracted.get(23)
                            r_hip = extracted.get(24)
                            l_sho = extracted.get(11)
                            r_sho = extracted.get(12)
                            
                            scale = 1.0
                            origin_x, origin_y = 0.0, 0.0
                            
                            if l_hip and r_hip and l_sho and r_sho:
                                origin_x = (l_hip['x'] + r_hip['x']) / 2.0
                                origin_y = (l_hip['y'] + r_hip['y']) / 2.0
                                mid_sho_x = (l_sho['x'] + r_sho['x']) / 2.0
                                mid_sho_y = (l_sho['y'] + r_sho['y']) / 2.0
                                scale = math.hypot(mid_sho_x - origin_x, mid_sho_y - origin_y)
                            else:
                                xs = [lm['x'] for lm in extracted.values()]
                                ys = [lm['y'] for lm in extracted.values()]
                                if xs and ys:
                                    min_x, max_x = min(xs), max(xs)
                                    min_y, max_y = min(ys), max(ys)
                                    origin_x, origin_y = min_x, min_y
                                    scale = max_y - min_y
                                    
                            if scale < 1.0:
                                scale = 1.0
                                
                            # Calculate MSE against YOLO normalized coords
                            mse = 0.0
                            count = 0
                            for idx, y_pt in f_data['yolo'].items():
                                if idx in extracted:
                                    nx = (extracted[idx]['x'] - origin_x) / scale
                                    ny = (extracted[idx]['y'] - origin_y) / scale
                                    mse += (nx - y_pt['x'])**2 + (ny - y_pt['y'])**2
                                    count += 1
                                    
                            if count > 0:
                                mse /= count
                            else:
                                mse = float('inf')
                                
                            if mse < min_mse:
                                min_mse = mse
                                best_lms = extracted
                                
                    if best_lms:
                        extracted_landmarks = best_lms
                        # We re-normalize the winning pose to write to CSV
                        l_hip = extracted_landmarks.get(23)
                        r_hip = extracted_landmarks.get(24)
                        l_sho = extracted_landmarks.get(11)
                        r_sho = extracted_landmarks.get(12)
                        
                        scale = 1.0
                        origin_x, origin_y = 0.0, 0.0
                        
                        if l_hip and r_hip and l_sho and r_sho:
                            origin_x = (l_hip['x'] + r_hip['x']) / 2.0
                            origin_y = (l_hip['y'] + r_hip['y']) / 2.0
                            mid_sho_x = (l_sho['x'] + r_sho['x']) / 2.0
                            mid_sho_y = (l_sho['y'] + r_sho['y']) / 2.0
                            scale = math.hypot(mid_sho_x - origin_x, mid_sho_y - origin_y)
                        else:
                            xs = [lm['x'] for lm in extracted_landmarks.values()]
                            ys = [lm['y'] for lm in extracted_landmarks.values()]
                            if xs and ys:
                                min_x, max_x = min(xs), max(xs)
                                min_y, max_y = min(ys), max(ys)
                                origin_x, origin_y = min_x, min_y
                                scale = max_y - min_y
                                
                        if scale < 1.0:
                            scale = 1.0
                            
                        for idx in sorted(landmark_mapping.keys()):
                            if idx in extracted_landmarks:
                                lm = extracted_landmarks[idx]
                                nx = (lm['x'] - origin_x) / scale
                                ny = (lm['y'] - origin_y) / scale
                                conf = lm['conf']
                                vals = [f"{nx:.3f}", f"{ny:.3f}", f"{conf:.3f}"]
                                row.extend(vals)
                                last_coords[idx] = vals
                            else:
                                row.extend(last_coords[idx])
                                
                        entity_landmarks = {}
                        for idx, pt in extracted_landmarks.items():
                            entity_landmarks[idx] = EntityLandmark(x=int(pt['x']), y=int(pt['y']), visibility=pt['conf'])
                        skel = Skeleton(landmarks=entity_landmarks)
                    else:
                        for idx in sorted(landmark_mapping.keys()):
                            row.extend(last_coords[idx])
                        skel = None
                            
                    writer.writerow(row)
                    
                    if enable_hud:
                        is_released = f_data['released'] == 'True'
                        active_side = f_data['handedness']
                        status_text = "RELEASED" if is_released else "PREPARE"
                        
                        angles = {"ls": None, "le": None, "lk": None, "rs": None, "re": None, "rk": None}
                        if skel:
                            if skel.left_shoulder_angle is not None: angles["ls"] = skel.left_shoulder_angle
                            if skel.left_elbow_angle is not None: angles["le"] = skel.left_elbow_angle
                            if skel.left_knee_angle is not None: angles["lk"] = skel.left_knee_angle
                            if skel.right_shoulder_angle is not None: angles["rs"] = skel.right_shoulder_angle
                            if skel.right_elbow_angle is not None: angles["re"] = skel.right_elbow_angle
                            if skel.right_knee_angle is not None: angles["rk"] = skel.right_knee_angle
                        
                        hud = HUD(
                            skeleton=skel,
                            status_text=status_text,
                            detector_name="MediaPipe",
                            active_side=active_side,
                            angles=angles
                        )
                        
                        mock_video = MockVideo()
                        if skel:
                            skel.draw(mock_video, frame)
                        hud.draw(mock_video, frame)
                        
                        video_writer.write(frame)
            
            if video_writer is not None:
                video_writer.release()
                video_writer = None
                
            throws_processed += 1
            
            # Free frame data between throws to keep memory low
            frame_data = None
            gc.collect()
            
            if not is_parallel:
                print(f"\nWrote {out_csv_path}!")
            else:
                safe_print(f"[{base_video_id}] Finished {unique_name}!")

    cap.release()
    total_time = time.perf_counter() - t_process_start
    return f"Processed {base_video_id}: {throws_processed} throws in {total_time:.1f}s"

def main():
    parser = argparse.ArgumentParser(description="Extract MediaPipe skeleton data for throws based on existing CSVs")
    parser.add_argument("--video", help="Video file of original run")
    parser.add_argument("--output_path", default="out/", help="Base path for input/output data (where existing CSVs are)")
    parser.add_argument("--data_dir", help="Data directory containing experiment/participant/runs folders to process everything automatically")
    parser.add_argument("--enable-hud", action="store_true", help="Enable rendering of debug videos with HUD overlays")
    parser.add_argument("--workers", type=int, default=2, help="Number of parallel workers for batch mode (default: 2, increase carefully based on available RAM)")
    args = parser.parse_args()

    if not args.video and not args.data_dir:
        print("Error: Must provide either --video or --data_dir")
        return

    # FULL MediaPipe indices (0 to 32)
    landmark_mapping = {
        0: "nose", 1: "left_eye_inner", 2: "left_eye", 3: "left_eye_outer", 
        4: "right_eye_inner", 5: "right_eye", 6: "right_eye_outer", 
        7: "left_ear", 8: "right_ear", 9: "mouth_left", 10: "mouth_right",
        11: "left_shoulder", 12: "right_shoulder", 13: "left_elbow", 14: "right_elbow",
        15: "left_wrist", 16: "right_wrist", 17: "left_pinky", 18: "right_pinky", 
        19: "left_index", 20: "right_index", 21: "left_thumb", 22: "right_thumb", 
        23: "left_hip", 24: "right_hip", 25: "left_knee", 26: "right_knee", 
        27: "left_ankle", 28: "right_ankle", 29: "left_heel", 30: "right_heel", 
        31: "left_foot_index", 32: "right_foot_index"
    }

    import urllib.request
    model_path = os.path.join(os.path.dirname(__file__), "pose_landmarker_heavy.task")
    if not os.path.exists(model_path):
        print("Downloading MediaPipe Pose model (only happens once)...")
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
            model_path
        )

    if args.data_dir:
        import threading
        manager = mp_lib.Manager()
        log_queue = manager.Queue()
        tasks = []
        for root, dirs, files in os.walk(args.data_dir):
            if "runs" in root.split(os.sep):
                for file in files:
                    if file.lower().endswith(('.mp4', '.mov')):
                        video_path = os.path.join(root, file)
                        runs_dir = os.path.dirname(video_path)
                        participant_dir = os.path.dirname(runs_dir)
                        output_path = os.path.join(participant_dir, "throw")
                        
                        if os.path.exists(output_path):
                            tasks.append((video_path, output_path, landmark_mapping, args.enable_hud, log_queue))
                            
        if not tasks:
            print("No matching videos found in data_dir.")
            return
            
        num_workers = min(args.workers, len(tasks))
        print(f"==================================================")
        print(f"Starting MediaPipe batch parallel processing!")
        print(f"Workers: {num_workers} (use --workers N to adjust)")
        print(f"Tasks queued: {len(tasks)}")
        print(f"==================================================")
        
        t_batch_start = time.perf_counter()
        
        try:
            from tqdm import tqdm
            has_tqdm = True
        except ImportError:
            has_tqdm = False
            
        def logger_thread(q, use_tqdm):
            while True:
                msg = q.get()
                if msg is None:
                    break
                if use_tqdm:
                    tqdm.write(msg)
                else:
                    print(msg)
                    
        log_thread = threading.Thread(target=logger_thread, args=(log_queue, has_tqdm))
        log_thread.start()
        
        # maxtasksperchild=1 ensures workers are recycled after each video,
        # freeing all leaked C++ memory from the MediaPipe runtime.
        with mp_lib.Pool(num_workers, maxtasksperchild=1) as pool:
            if has_tqdm:
                for res in tqdm(pool.imap_unordered(process_video_worker, tasks), total=len(tasks), desc="Extracting MediaPipe (Parallel)"):
                    tqdm.write(res)
            else:
                for idx, res in enumerate(pool.imap_unordered(process_video_worker, tasks)):
                    print(f"[{idx+1:03d}/{len(tasks):03d}] {res}")
                    
        log_queue.put(None)
        log_thread.join()
                
        total_batch_time = time.perf_counter() - t_batch_start
        print(f"\nBatch processing finished in {total_batch_time:.1f} seconds!")
        
    else:
        res = process_video(args.video, args.output_path, landmark_mapping, args.enable_hud, is_parallel=False)
        print(res)
        print("==================================================")
        print("MediaPipe Processing Done!")

if __name__ == "__main__":
    main()
