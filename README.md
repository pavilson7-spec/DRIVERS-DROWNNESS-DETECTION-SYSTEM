📁 Project Structure
driver-drowsiness-detection/
│
├── assets/
│ └── alarm.mp3
│
├── main.py
├── requirements.txt
└── README.md

Driver Drowsiness Detection System

A real-time **Driver Drowsiness Detection System** built using Computer Vision and AI to enhance road safety by detecting driver fatigue and triggering alerts.

Overview

This project uses **MediaPipe FaceMesh** to detect facial landmarks and calculate the **Eye Aspect Ratio (EAR)**.  
If the driver’s eyes remain closed for a certain duration, the system identifies drowsiness and plays an alarm sound.

Features

Real-time webcam monitoring  
Eye tracking using FaceMesh  
Eye Aspect Ratio (EAR) calculation  
Instant drowsiness alert system  
Alarm sound using Pygame  
Multi-threaded alert mechanism  

Tech Stack

- Python  
- OpenCV  
- MediaPipe  
- SciPy  
- Pygame  

 How It Works

1. Capture live video using webcam  
2. Detect face landmarks using MediaPipe  
3. Extract eye coordinates  
4. Calculate EAR (Eye Aspect Ratio)  
5. Check if EAR < threshold  
6. If condition persists → Trigger alarm 🚨  

 EAR Formula

EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)

Install dependencies:
pip install -r requirements.txt

Run the project:
python main.py
<img width="1262" height="643" alt="image" src="https://github.com/user-attachments/assets/8bef0661-ea75-42f9-a12e-9d922b17788f" />
