from flask import Flask, Response, jsonify
from flask_cors import CORS  # Enable CORS for cross-origin requests
import threading
import cv2
from ultralytics import YOLO
import atexit  # Ensures resources are released when the app exits
import time

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS

# Global variables
alert_message = "No threats detected."
lock = threading.Lock()  # Thread-safe lock for shared resources
camera_status = "offline"  # Start with offline status

# Load YOLOv8 model (pre-trained)
model = YOLO("best.pt").to("cuda")  # Load your trained model

# Open the camera in a separate thread to ensure it's ready
def initialize_camera():
    global cap, camera_status
    cap = cv2.VideoCapture(0)  # Change to 1 or 2 if using an external camera

    retry_attempts = 5
    while retry_attempts > 0:
        if cap.isOpened():
            # Warm-up the camera
            for _ in range(10):
                ret, _ = cap.read()
                if ret:
                    camera_status = "online"
                    print("Camera initialized and ready.")
                    return
                time.sleep(0.1)
        retry_attempts -= 1
        time.sleep(1)

    print("Camera failed to open. Check if another application is using it.")
    camera_status = "offline"

# Start camera initialization in a background thread
camera_thread = threading.Thread(target=initialize_camera)
camera_thread.start()

# Function to generate video frames with YOLO detection
def generate_frames():
    global alert_message, camera_status

    while True:
        if camera_status == "offline":
            break

        ret, frame = cap.read()
        if not ret or frame is None:
            camera_status = "offline"
            break

        # Run YOLO object detection
        results = model(frame)

        # Check for threats (cell phone, violence, guns, weapons)
        with lock:
            alert_message = "No threats detected."
            for result in results:
                for box in result.boxes:
                    label = model.model.names[int(box.cls[0])]  # Fix label extraction
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Set bounding box colors based on severity
                    if label in ["gun", "weapon", "violence"]:
                        color = (0, 0, 255)  # Red for critical threats
                        alert_message = f"{label.capitalize()} detected!"
                    else:
                        color = (255, 0, 0)  # Blue for non-critical (e.g., cell phone)

                    # Draw bounding box & label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(frame, label.upper(), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Stream frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n'
               b'Content-Length: ' + f"{len(frame_bytes)}".encode() + b'\r\n\r\n' +
               frame_bytes + b'\r\n')

# Flask route to stream video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# API endpoint to get the latest alert
@app.route('/api/alert', methods=['GET'])
def get_alert():
    with lock:
        return jsonify({"message": alert_message, "camera_status": camera_status})

# Health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    with lock:
        return jsonify({"camera_status": camera_status})

# Flask route for the home page
@app.route('/')
def home():
    return '''
    <html>
        <head><title>YOLOv8 CCTV Detection</title></head>
        <body>
            <h1>YOLOv8 CCTV Detection Running...</h1>
            <p>Check the <a href="/video_feed">Live Feed</a></p>
        </body>
    </html>
    '''

# Release camera on exit
@atexit.register
def release_camera():
    print("Releasing camera resources...")
    if 'cap' in globals():
        cap.release()
    cv2.destroyAllWindows()

# Run Flask app
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)
