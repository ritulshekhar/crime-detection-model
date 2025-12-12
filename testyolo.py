import cv2
import torch
import os
from ultralytics import YOLO

# ✅ Fix OpenCV GUI issue (for Windows users)
os.environ["QT_QPA_PLATFORM"] = "windows"  

# ✅ Check CUDA Availability
print("CUDA Available:", torch.cuda.is_available())

model = YOLO("best.pt")  # Change to "yolov8n.pt" if testing pre-trained model

cap = cv2.VideoCapture(0)  

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# ✅ Set Camera Properties (Optional)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Run YOLO detection
    results = model(frame)  
    annotated_frame = results[0].plot()  # Overlay detections

    # ✅ Display Output Window
    cv2.imshow("YOLOv8 Webcam", annotated_frame)
    
    # ✅ Press 'q' to Exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ✅ Release Resources
cap.release()
cv2.destroyAllWindows()
