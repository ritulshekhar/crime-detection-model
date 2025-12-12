AI-Powered Crime Detection Model

A real-time AI surveillance system designed to detect weapons and suspicious objects using YOLOv8, with a scalable Flask backend, MongoDB database, and an interactive React.js dashboard for monitoring cameras and receiving alerts.

🚀 Project Overview

The AI-Powered Crime Detection Model is an end-to-end surveillance intelligence system that performs:

Real-time weapon detection using YOLOv8 (Ultralytics)

REST API–based backend using Flask + PyTorch

Camera management & alert logging via MongoDB

Live monitoring dashboard built with React.js

Scalable microservice-friendly architecture

This project demonstrates strong expertise in deep learning, computer vision, backend engineering, frontend development, and database design.

📌 Key Features
1. Real-Time Crime & Weapon Detection

YOLOv8 trained/fine-tuned to detect weapons (guns, knives, etc.)

Frame-by-frame inference with optimized TorchVision transformations

Configurable confidence thresholds

2. Flask Backend for Model Serving

REST API endpoints for:

Uploading frames / video streams

Registering and managing cameras

Sending and retrieving alert logs

Modular structure for easy scaling and containerization

3. MongoDB Persistence Layer

Collections for:

Cameras (IDs, IPs, locations, status)

Alerts (timestamp, weapon type, camera source, confidence scores)

Optimized schema for fast read/write operations

4. React.js Monitoring Dashboard

Live alert panel with real-time updates

Dynamic camera list: add / remove / update camera streams

Visual overlays for detected objects

Built for operational teams to quickly respond to threats

5. Agile & Collaborative Development

Designed with modularity to support multi-team development

Clear API contracts for frontend ↔ backend integration

🛠️ Tech Stack
Machine Learning

YOLOv8

PyTorch

TorchVision

Ultralytics

Backend

Python

Flask (REST APIs)

Frontend

React.js

Axios

Tailwind / CSS Modules (optional)

Database

MongoDB

PyMongo

⚙️ Installation & Setup
1. Clone the Repository
git clone https://github.com/your-username/crime-detection-model.git
cd crime-detection-model

📌 Backend Setup (Flask + YOLOv8)
Create Virtual Environment
cd backend
python3 -m venv venv
source venv/bin/activate

Install Dependencies
pip install -r requirements.txt

Run Flask Server
python app.py


Backend runs on:

http://localhost:5000

🖥️ Frontend Setup (React.js Dashboard)
cd frontend
npm install
npm start


Frontend runs on:

http://localhost:3000

📡 API Endpoints Overview
Camera Management
Method	Endpoint	Description
POST	/api/cameras/add	Register a new camera
GET	/api/cameras/	Fetch all cameras
DELETE	/api/cameras/<id>	Remove camera
Alert Management
Method	Endpoint	Description
POST	/api/alert	Log an alert
GET	/api/alert/all	Fetch all alerts
Inference
Method	Endpoint	Description
POST	/api/detect	Upload frame → YOLO detection
📈 Detection Flow

Camera sends frame → Flask API /api/detect

YOLOv8 processes frame and returns:

Detected weapon type

Confidence score

Bounding boxes

If threat detected:

Alert stored in MongoDB

React dashboard receives live update

📦 Model Training (Optional)

If training custom weights:

yolo detect train data=data.yaml model=yolov8n.pt epochs=50 imgsz=640

🧪 Testing
python testyolo.py

🔒 Future Improvements

Add person identification using Re-ID models

Multi-camera fusion

SMS/Email/WhatsApp alert automation

On-device edge inference (Jetson Nano / Coral TPU)

Role-based admin dashboard
