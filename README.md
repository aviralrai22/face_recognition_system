# face_recognition_system
# 👤 Real-Time Face Recognition System

A Python-based computer vision project that detects and recognizes faces in real-time using the Local Binary Patterns Histograms (LBPH) algorithm.

## 🚀 Features
- **Data Collection:** Captures face samples via webcam and assigns unique IDs.
- **Model Training:** Trains an LBPH recognizer on the captured dataset.
- **Real-Time Recognition:** Identifies faces in a live video stream with confidence scores.
- **Robustness:** Handles basic lighting variations and angles.

## 🛠️ Tech Stack
- **Language:** Python 3.x
- **Libraries:** OpenCV (`opencv-contrib-python`), NumPy, Pillow (PIL), OS

## 📂 Project Structure
```text
├── dataset/                   # Stores captured face images (User.ID.Count.jpg)
├── trainer/                   # Stores the trained model (trainer.yml)
├── data_collection.py         # Script 1: Capture face samples
├── train_model.py             # Script 2: Train the LBPH recognizer
├── main_recognition.py        # Script 3: Run the live recognition system
├── haarcascade_frontalface_default.xml # (Optional if not using cv2.data)
└── README.md
