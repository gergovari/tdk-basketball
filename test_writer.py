import cv2
print("Testing VideoWriter with fps=0.0")
out = cv2.VideoWriter("test_zero_fps.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 0.0, (100, 100))
print("isOpened?", out.isOpened())
import numpy as np
out.write(np.zeros((100, 100, 3), dtype=np.uint8))
out.release()
import os
print("File size:", os.path.getsize("test_zero_fps.mp4") if os.path.exists("test_zero_fps.mp4") else "Not found")
