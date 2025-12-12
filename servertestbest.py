import cv2
import time
import torch
import os
from ultralytics import YOLO

# ✅ Fix OpenCV GUI issue (for Windows users)
os.environ["QT_QPA_PLATFORM"] = "windows"

# ✅ Load YOLO Model
model = YOLO("best.pt").to("cuda")  # Change to correct model file

# ✅ Open Webcam (0 for default, 1/2 for external)
cap = cv2.VideoCapture(0)

# ✅ Check if Webcam is Opened
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# ✅ Display CUDA Availability
print(f"CUDA Available: {torch.cuda.is_available()}")

while True:
    start_time = time.perf_counter()  # High-precision timer
    
    success, frame = cap.read()
    if not success:
        print("Error: Failed to capture image.")
        break

    # ✅ Run YOLO Inference
    results = model(frame, imgsz=640, conf=0.4)  
    annotated_frame = results[0].plot()  

    # ✅ Display FPS Calculation
    end_time = time.perf_counter()
    inference_time = end_time - start_time
    fps = 1 / inference_time if inference_time > 0 else 0
    print(f"Inference Time: {inference_time:.3f}s, FPS: {fps:.2f}")

    # ✅ Show Output Window
    cv2.imshow("YOLOv8 Webcam", annotated_frame)

    # ✅ Exit on 'q' Key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ✅ Release Resources
cap.release()
cv2.destroyAllWindows()
