import cv2
from video import Video
v = Video("dummy_run/dummy.mp4", "test_out.mp4")
print("fps before release:", v.fps)
v.release()
print("fps after release:", v.fps)
