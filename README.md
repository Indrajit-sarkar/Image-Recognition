# 🔍 ADVANCED MULTI-MODAL IMAGE RECOGNITION SYSTEM

A comprehensive, high-performance image recognition system with cloud API integration, real-time object detection, currency recognition, people detection, and gesture-based cursor control. Built with Python using YOLOv8, MediaPipe, Google Cloud Vision, and Roboflow APIs.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-red.svg)
![License](https://img.shields.io/badge/License-Educational-yellow.svg)

---

## 📑 TABLE OF CONTENTS

1. [Features Overview](#-features-overview)
2. [System Requirements](#-system-requirements)
3. [Installation Guide](#-installation-guide)
4. [API Configuration](#-api-configuration)
5. [Usage Instructions](#-usage-instructions)
6. [Gesture Controls](#-gesture-controls)
7. [API Characteristics](#-api-characteristics)
8. [Project Structure](#-project-structure)
9. [Troubleshooting](#-troubleshooting)
10. [License](#-license)

---

## ✨ FEATURES OVERVIEW

### Core Detection Capabilities

| Feature | Description |
|---------|-------------|
| **Object Detection** | Detect 80+ object categories using YOLOv8 with cloud verification |
| **Currency Detection** | Identify banknotes and coins from 8 currencies (USD, EUR, GBP, INR, JPY, CNY, AUD, CAD) |
| **People Detection** | Optimized person detection with hybrid local+cloud mode |
| **Face & Eye Tracking** | Real-time face detection with eye tracking capabilities |
| **Text Recognition (OCR)** | Multi-language text detection and recognition |
| **Hand Gesture Control** | Control your computer cursor using hand gestures |
| **Depth Estimation** | Estimate distances to detected objects |
| **Medicine Detection** | Identify pills and medicine bottles |
| **Food Detection** | Recognize various food items |

### Advanced Features

- 🔄 **Hybrid Detection** - Combines local YOLO models with cloud APIs for best accuracy
- 🌐 **Multi-API Support** - Google Cloud Vision, Roboflow, and Clarifai integration
- 📊 **Kalman Filter Smoothing** - Ultra-smooth cursor control with advanced filtering
- ⚡ **Real-time Processing** - 30+ FPS on GPU-enabled systems
- 🔀 **Fallback Mechanisms** - Automatic fallback to backup models if primary fails
- 💾 **Caching System** - Intelligent caching to reduce API calls and improve performance

---

## 💻 SYSTEM REQUIREMENTS

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Processor | Intel i5 / AMD Ryzen 5 | Intel i7 / AMD Ryzen 7 or higher |
| RAM | 8 GB | 16 GB or higher |
| GPU | Not required (CPU mode) | NVIDIA GPU with CUDA support (4GB+ VRAM) |
| Camera | 720p webcam | 1080p webcam or higher |
| Storage | 2 GB free space | 5 GB free space |

### Software Requirements

| Software | Version | Purpose |
|----------|---------|---------|
| Python | 3.9 - 3.11 | Runtime environment |
| Windows | 10 / 11 | Operating system |
| CUDA (Optional) | 11.8+ | GPU acceleration |
| cuDNN (Optional) | 8.6+ | Deep learning optimization |

### Python Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| torch | ≥2.0.0 | Deep learning framework |
| torchvision | ≥0.15.0 | Computer vision models |
| ultralytics | ≥8.0.0 | YOLOv8 object detection |
| opencv-python | ≥4.8.0 | Image processing |
| mediapipe | ≥0.10.0 | Hand, face, pose detection |
| google-cloud-vision | ≥3.5.0 | Google Vision API client |
| roboflow | ≥1.1.0 | Roboflow API client |
| easyocr | ≥1.7.0 | Text recognition |
| pyautogui | ≥0.9.54 | Mouse/keyboard control |
| python-dotenv | ≥1.0.0 | Environment variable management |

---

## 📦 INSTALLATION GUIDE

### Step 1: Clone or Download the Project

```bash
git clone https://github.com/yourusername/image-recognition.git
cd image-recognition
```

Or download the ZIP file and extract it to your desired location.

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows)
venv\Scripts\activate

# Activate virtual environment (Linux/Mac)
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Or use py command on Windows if pip doesn't work
py -m pip install -r requirements.txt
```

### Step 4: Configure API Keys

```bash
# Copy the environment template
copy .env.example .env

# Edit .env file and add your API keys
notepad .env
```

### Step 5: Verify Installation

```bash
python main.py --help
```

---

## 🔑 API CONFIGURATION

### Google Cloud Vision API Setup

**Step 1:** Visit [Google Cloud Console](https://console.cloud.google.com) and create an account. New users get **$300 in free credits**.

**Step 2:** Create a new project named "ImageRecognition"

**Step 3:** Navigate to APIs & Services → Library → Search "Cloud Vision API" → Enable

**Step 4:** Go to APIs & Services → Credentials → Create Credentials → API Key

**Step 5:** Add to `.env` file:
```
GOOGLE_VISION_API_KEY=your_api_key_here
```

### Roboflow API Setup

**Step 1:** Visit [Roboflow](https://roboflow.com) and create a free account

**Step 2:** Go to Settings → API → Copy your Private API Key

**Step 3:** Add to `.env` file:
```
ROBOFLOW_API_KEY=your_roboflow_key_here
```

### Clarifai API Setup (Optional)

**Step 1:** Visit [Clarifai](https://clarifai.com) and sign up

**Step 2:** Go to Settings → Security → Create Personal Access Token

**Step 3:** Add to `.env` file:
```
CLARIFAI_API_KEY=your_clarifai_key_here
```

---

## 🚀 USAGE INSTRUCTIONS

### Real-Time Camera Detection

```bash
# Start real-time detection with default camera
python main.py --mode realtime

# Use specific camera (0=built-in, 1=external USB camera)
python main.py --mode realtime --camera 1
```

### Process Single Image

```bash
python main.py --mode image --input path/to/your/image.jpg
```

### Process Video File

```bash
python main.py --mode video --input path/to/your/video.mp4
```

### Keyboard Controls

| Key | Action |
|-----|--------|
| `Q` | Quit the application |
| `M` | Toggle mode (Detection → Cursor → 3D) |
| `C` | Create cube (3D mode) |
| `P` | Create pyramid (3D mode) |
| `S` | Create sphere (3D mode) |
| `Y` | Create cylinder (3D mode) |
| `X` | Clear all 3D structures |

---

## 🖐️ GESTURE CONTROLS

### Available Gestures (Cursor Mode)

| Gesture | Description | Action |
|---------|-------------|--------|
| ☝️ **Pointing** | Index finger extended, others closed | Move cursor |
| 🤏 **Pinch** | Thumb and index finger touch | Left click |
| 🤏🤏 **Double Pinch** | Pinch twice quickly | Double click |
| 🤌 **Three-Finger Pinch** | Thumb, index, and middle touch | Right click |
| ✌️ **Peace Sign** | Index and middle fingers up | Scroll mode |
| ✌️↕️ **Peace + Move** | Move peace sign vertically | Scroll up/down |
| ✊ **Fist** | All fingers closed | Drag mode |
| 🖐️ **Open Palm** | All fingers extended | Release/Stop |
| 👍 **Thumbs Up** | Thumb up, others closed | Confirm action |
| 👎 **Thumbs Down** | Thumb down, others closed | Cancel action |

### Gesture Tips

- Keep your hand **30-60 cm** from the camera for best detection
- Ensure **good lighting** for accurate gesture recognition
- Move **slowly and deliberately** when performing gestures
- The cursor uses **Kalman filter smoothing** for stability

---

## 📊 API CHARACTERISTICS

### Google Cloud Vision API

| Feature | Details |
|---------|---------|
| Free Tier | 1,000 units per month |
| New User Credits | $300 free credits for 90 days |
| Supported Features | Object Localization, Text Detection (OCR), Face Detection, Label Detection |
| Response Time | 1-3 seconds per image |
| Image Size Limit | 20 MB maximum |
| Rate Limit | 1,800 requests per minute |
| Accuracy | High accuracy for common objects, excellent OCR |

### Roboflow API

| Feature | Details |
|---------|---------|
| Free Tier | 1,000 inference credits per month |
| Specialization | Custom trained models for specific objects |
| Response Time | 0.5-2 seconds per image |
| Model Hosting | Host and deploy custom trained models |
| Accuracy | Custom models can achieve 90%+ accuracy |

### Clarifai API (Optional)

| Feature | Details |
|---------|---------|
| Free Tier | 1,000 operations per month |
| Features | General image recognition, concept detection |
| Response Time | 1-2 seconds per image |
| Pre-trained Models | Food, apparel, travel, and more |

### Local Detection (YOLOv8)

| Feature | Details |
|---------|---------|
| Models Included | yolov8n.pt (fast), yolov8x.pt (accurate) |
| Object Classes | 80 COCO dataset classes |
| Speed (GPU) | 30-60 FPS with NVIDIA GPU |
| Speed (CPU) | 5-15 FPS depending on processor |
| Offline Capable | Yes, works without internet |

### Hybrid Detection Strategy

1. **Local First** - YOLOv8 processes frames locally for speed
2. **Cloud Verification** - Important detections verified with cloud APIs
3. **Automatic Fallback** - If cloud APIs unavailable, local detection continues
4. **Result Caching** - Cloud results cached for 1 second
5. **Rate Limiting** - Built-in rate limiting to stay within free tiers

---

## 📁 PROJECT STRUCTURE

```
Image Recognition/
├── main.py                      # Main application entry point
├── config.yaml                  # Configuration settings
├── requirements.txt             # Python dependencies
├── .env.example                 # API key template
├── .env                         # Your API keys (create from .env.example)
├── .gitignore                   # Git ignore rules
├── README.md                    # This file
│
├── detectors/                   # Detection modules
│   ├── __init__.py
│   ├── object_detector.py       # YOLOv8 + cloud object detection
│   ├── cloud_vision_detector.py # Unified cloud API wrapper
│   ├── currency_detector.py     # Currency note/coin detection
│   ├── hand_gesture_detector.py # Hand gesture recognition
│   ├── face_eye_tracker.py      # Face and eye detection
│   ├── ocr_engine.py            # Text recognition
│   ├── depth_estimator.py       # Distance estimation
│   ├── medicine_detector.py     # Medicine identification
│   └── food_detector.py         # Food recognition
│
├── utils/                       # Utility modules
│   ├── __init__.py
│   ├── api_key_manager.py       # Secure API key handling
│   ├── cursor_controller.py     # Mouse control with Kalman filter
│   ├── camera_manager.py        # Camera handling
│   ├── visualization.py         # Drawing and display
│   ├── result_aggregator.py     # Combine detection results
│   ├── performance_optimizer.py # GPU/CPU optimization
│   └── structure_3d_creator.py  # 3D object creation
│
├── results/                     # Output directory
│
├── yolov8n.pt                   # YOLOv8 Nano model (fast)
└── yolov8x.pt                   # YOLOv8 Extra-large model (accurate)
```

---

## 🔧 TROUBLESHOOTING

### Common Issues

| Problem | Solution |
|---------|----------|
| pip install fails | Use `py -m pip install -r requirements.txt` |
| Camera not detected | Try different camera index: `--camera 1` or `--camera 2` |
| CUDA out of memory | Use smaller model: change `yolov8x` to `yolov8n` in config.yaml |
| Cloud API not working | Check .env file has correct API keys; verify API is enabled |
| Slow performance | Disable cloud APIs in config.yaml or use GPU acceleration |
| Hand gestures not detected | Improve lighting; keep hand 30-60cm from camera |

### Error Messages

**"No API key found for google_vision"**
> Create .env file from .env.example and add your Google Vision API key.

**"Failed to load primary model"**
> Ensure yolov8x.pt or yolov8n.pt files exist in the project directory.

**"Camera read failed"**
> Check if another application is using the camera. Try restarting.

---

## 📄 LICENSE

This project is provided for educational and personal use. Commercial use may require additional licensing for:

- **YOLOv8** - AGPL-3.0 License (Ultralytics)
- **MediaPipe** - Apache 2.0 License (Google)
- **Google Cloud Vision** - Subject to Google Cloud Terms of Service
- **Roboflow** - Subject to Roboflow Terms of Service

---

<div align="center">

### 🔍 Advanced Multi-Modal Image Recognition System

**Version 2.0** | February 2026

Built with Python, YOLOv8, MediaPipe, and Cloud AI

⭐ Star this repo if you find it useful!

</div>
