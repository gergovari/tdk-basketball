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
)

# params = InputParams(video_id="ft1_v108_002351_x264", data_path="data/")
params = InputParams(video_id="nba1", data_path="data/")
player_filter = ["player", "person", "human"]
ball_filter = ["ball"]
yolo_filter = player_filter + ball_filter
yolo_params = YOLOParams(model_path="models/basketball-3-m.pt", name_filter=yolo_filter)
mp_params = MediaPipeParams(
    model_path="models/pose_landmarker.task", min_pose_conf=0, min_track_conf=0
)

model = YOLO(yolo_params.model_path)
yolo_filtered = YOLOFiltered(model, yolo_params.name_filter)

mediapipe = MediaPipe(mp_params)

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
    print("No valid thrower detected for this clip!")
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
    print("Release not detected!")
print("Cut!")

print(f"Exporting skeleton data to {params.output_data_path}...")
export_skeleton_data(obj_frames, params.output_data_path, video.fps, release_frame)
print("Exported!")

print(f"Writing out to {params.output_video_path}...")
render_video(video, obj_frames, release_frame, release_detector_name)
print("Wrote!")

video.release()

print("Done!")
