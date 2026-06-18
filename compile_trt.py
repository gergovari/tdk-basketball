"""Standalone script to compile the TensorRT engine from .pt weights."""
from ultralytics import YOLO

print("Loading YOLOv8x-pose PyTorch model...")
model = YOLO("models/yolov8x-pose-p6.pt")

print("Compiling to TensorRT engine (this may take 5-15 minutes depending on GPU)...")
model.export(format="engine", half=True, dynamic=False, imgsz=320, device="cuda")
print("Compilation finished! The .engine file should be in the models directory.")
