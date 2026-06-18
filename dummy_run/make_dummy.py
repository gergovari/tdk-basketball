import cv2
import numpy as np
out = cv2.VideoWriter("dummy.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (1280, 720))
for i in range(150):
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.putText(img, f"Frame {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    out.write(img)
out.release()
