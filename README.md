# 😊 Real-Time Emotion Detection System
A Python-based **Real-Time Emotion Detection System** that uses Computer Vision, OpenCV, and DeepFace to analyze facial expressions through a webcam and identify human emotions.

The system captures live webcam footage, detects faces, analyzes facial expressions, and displays the predicted emotion in real time.

## 🚀 Features

- **Real-Time Webcam Detection:** Captures live video using OpenCV.
- **Face Detection:** Detects human faces using Haar Cascade Classifier.
- **Emotion Recognition:** Uses DeepFace and a pretrained emotion recognition model.
- **Seven Emotion Categories:** Recognizes:
  - Happy
  - Sad
  - Angry
  - Fear
  - Surprise
  - Disgust
  - Neutral
- **Live Emotion Display:** Displays the predicted emotion on the webcam feed.
- **Automatic Processing:** Continuously analyzes video frames.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.11 | Programming Language |
| OpenCV | Webcam access and face detection |
| DeepFace | Facial emotion recognition |
| TensorFlow | Deep Learning Framework |
| TF-Keras | Model compatibility |

## 📁 Project Structure

```text
Real-Time-Emotion-Detection/
│
├── emotion_detector.py
├── README.md
└── emotion-env/        # Local virtual environment
```

*Note: Do not upload your virtual environment (`emotion-env`) to GitHub.*

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/Real-Time-Emotion-Detection.git
```

### 2. Navigate to the Project Directory

```bash
cd Real-Time-Emotion-Detection
```

### 3. Create a Virtual Environment

```bash
py -3.11 -m venv emotion-env
```

### 4. Activate the Environment

**Windows:**

```bash
emotion-env\Scripts\activate
```

### 5. Install Required Libraries

```bash
python -m pip install --upgrade pip
```

```bash
python -m pip install opencv-python deepface tf-keras
```

### 6. Run the Project

```bash
python emotion_detector.py
```

Your webcam will open and start the emotion detection process.

Press **ESC** to exit the application.

## 🧠 How It Works

1. The webcam captures real-time video using OpenCV.
2. Each frame is processed for facial detection.
3. The detected facial information is analyzed using DeepFace.
4. The pretrained emotion recognition model predicts the dominant facial expression.
5. The predicted emotion is displayed on the webcam window.

## 📸 Expected Output

The application displays a live webcam feed with:
- Green bounding boxes around detected faces.
- Predicted emotion labels.
- Continuous emotion analysis.

## 🔮 Future Improvements

- Emotion detection for multiple faces independently.
- Emotion confidence percentage display.
- Emotion-based bounding box colours.
- Live emotion statistics dashboard.
- Graphical user interface (GUI).
- Emotion history and data visualization.

## 🎯 Project Objective

The objective of this project is to develop a real-time facial emotion detection application using computer vision and deep learning techniques. It demonstrates how artificial intelligence can be used to analyze facial expressions and classify emotions from live video input.

## ⚠️ Limitations

- Emotion predictions are estimates based on visible facial expressions and do not reveal a person's actual feelings.
- Detection accuracy can vary depending on lighting, camera quality, and facial positioning.
- Real-time performance depends on system hardware.

## 👨‍💻 Author

**Amitesh Jha**

Computer Science & Engineering Student

GitHub: [Your GitHub Profile](https://github.com/YOUR-USERNAME)

---

⭐ If you find this project interesting, consider giving the repository a star!
