import cv2
cap = cv2.VideoCapture("out.mp4") # Assume a valid video or we'll make one
# let's write a small video first
writer = cv2.VideoWriter("test.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (100, 100))
import numpy as np
for i in range(10): writer.write(np.zeros((100, 100, 3), dtype=np.uint8))
writer.release()

from video import Video
v = Video("test.mp4", "test_out.mp4")
print("fps before release:", v.fps)
v.release()
try:
    print("fps after release:", v.fps)
except Exception as e:
    print("exception:", e)
