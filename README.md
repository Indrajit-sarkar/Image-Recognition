<html>
<head>
<style>
body {
    font-family: 'Times New Roman', Times, serif;
}
h1 {
    font-family: 'Times New Roman', Times, serif;
    font-size: 18px;
    font-weight: bold;
}
h2 {
    font-family: 'Times New Roman', Times, serif;
    font-size: 14px;
    font-weight: bold;
}
p, li, td, th {
    font-family: 'Times New Roman', Times, serif;
    font-size: 12px;
}
code {
    font-family: 'Courier New', monospace;
    font-size: 11px;
    background-color: #f4f4f4;
    padding: 2px 4px;
}
pre {
    font-family: 'Courier New', monospace;
    font-size: 11px;
    background-color: #f4f4f4;
    padding: 10px;
    border-radius: 5px;
}
table {
    border-collapse: collapse;
    width: 100%;
    margin: 10px 0;
}
th, td {
    border: 1px solid #ddd;
    padding: 8px;
    text-align: left;
}
th {
    background-color: #4a90d9;
    color: white;
}
</style>
</head>
<body>

<h1>ADVANCED MULTI-MODAL IMAGE RECOGNITION SYSTEM</h1>

<p>A comprehensive, high-performance image recognition system with cloud API integration, real-time object detection, currency recognition, people detection, and gesture-based cursor control. Built with Python using YOLOv8, MediaPipe, Google Cloud Vision, and Roboflow APIs.</p>

---

<h1>TABLE OF CONTENTS</h1>

<p>
1. Features Overview<br>
2. System Requirements<br>
3. Installation Guide<br>
4. API Configuration<br>
5. Usage Instructions<br>
6. Gesture Controls<br>
7. API Characteristics<br>
8. Project Structure<br>
9. Troubleshooting<br>
10. License
</p>

---

<h1>1. FEATURES OVERVIEW</h1>

<h2>1.1 Core Detection Capabilities</h2>

<p>
• <b>Object Detection</b> - Detect 80+ object categories using YOLOv8 with cloud verification<br>
• <b>Currency Detection</b> - Identify banknotes and coins from 8 currencies (USD, EUR, GBP, INR, JPY, CNY, AUD, CAD)<br>
• <b>People Detection</b> - Optimized person detection with hybrid local+cloud mode<br>
• <b>Face & Eye Tracking</b> - Real-time face detection with eye tracking capabilities<br>
• <b>Text Recognition (OCR)</b> - Multi-language text detection and recognition<br>
• <b>Hand Gesture Control</b> - Control your computer cursor using hand gestures<br>
• <b>Depth Estimation</b> - Estimate distances to detected objects<br>
• <b>Medicine Detection</b> - Identify pills and medicine bottles<br>
• <b>Food Detection</b> - Recognize various food items
</p>

<h2>1.2 Advanced Features</h2>

<p>
• <b>Hybrid Detection</b> - Combines local YOLO models with cloud APIs for best accuracy<br>
• <b>Multi-API Support</b> - Google Cloud Vision, Roboflow, and Clarifai integration<br>
• <b>Kalman Filter Smoothing</b> - Ultra-smooth cursor control with advanced filtering<br>
• <b>Real-time Processing</b> - 30+ FPS on GPU-enabled systems<br>
• <b>Fallback Mechanisms</b> - Automatic fallback to backup models if primary fails<br>
• <b>Caching System</b> - Intelligent caching to reduce API calls and improve performance
</p>

---

<h1>2. SYSTEM REQUIREMENTS</h1>

<h2>2.1 Hardware Requirements</h2>

<table>
<tr><th>Component</th><th>Minimum</th><th>Recommended</th></tr>
<tr><td>Processor</td><td>Intel i5 / AMD Ryzen 5</td><td>Intel i7 / AMD Ryzen 7 or higher</td></tr>
<tr><td>RAM</td><td>8 GB</td><td>16 GB or higher</td></tr>
<tr><td>GPU</td><td>Not required (CPU mode)</td><td>NVIDIA GPU with CUDA support (4GB+ VRAM)</td></tr>
<tr><td>Camera</td><td>720p webcam</td><td>1080p webcam or higher</td></tr>
<tr><td>Storage</td><td>2 GB free space</td><td>5 GB free space</td></tr>
</table>

<h2>2.2 Software Requirements</h2>

<table>
<tr><th>Software</th><th>Version</th><th>Purpose</th></tr>
<tr><td>Python</td><td>3.9 - 3.11</td><td>Runtime environment</td></tr>
<tr><td>Windows</td><td>10 / 11</td><td>Operating system</td></tr>
<tr><td>CUDA (Optional)</td><td>11.8+</td><td>GPU acceleration</td></tr>
<tr><td>cuDNN (Optional)</td><td>8.6+</td><td>Deep learning optimization</td></tr>
</table>

<h2>2.3 Python Dependencies</h2>

<p>The following packages are required (install via requirements.txt):</p>

<table>
<tr><th>Package</th><th>Version</th><th>Purpose</th></tr>
<tr><td>torch</td><td>≥2.0.0</td><td>Deep learning framework</td></tr>
<tr><td>torchvision</td><td>≥0.15.0</td><td>Computer vision models</td></tr>
<tr><td>ultralytics</td><td>≥8.0.0</td><td>YOLOv8 object detection</td></tr>
<tr><td>opencv-python</td><td>≥4.8.0</td><td>Image processing</td></tr>
<tr><td>mediapipe</td><td>≥0.10.0</td><td>Hand, face, pose detection</td></tr>
<tr><td>google-cloud-vision</td><td>≥3.5.0</td><td>Google Vision API client</td></tr>
<tr><td>roboflow</td><td>≥1.1.0</td><td>Roboflow API client</td></tr>
<tr><td>easyocr</td><td>≥1.7.0</td><td>Text recognition</td></tr>
<tr><td>pyautogui</td><td>≥0.9.54</td><td>Mouse/keyboard control</td></tr>
<tr><td>python-dotenv</td><td>≥1.0.0</td><td>Environment variable management</td></tr>
</table>

---

<h1>3. INSTALLATION GUIDE</h1>

<h2>3.1 Clone or Download the Project</h2>

<pre>
git clone https://github.com/yourusername/image-recognition.git
cd image-recognition
</pre>

<p>Or download the ZIP file and extract it to your desired location.</p>

<h2>3.2 Create Virtual Environment (Recommended)</h2>

<pre>
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On Linux/Mac:
source venv/bin/activate
</pre>

<h2>3.3 Install Dependencies</h2>

<pre>
# Install all required packages
pip install -r requirements.txt

# Or use py command on Windows if pip doesn't work
py -m pip install -r requirements.txt
</pre>

<h2>3.4 Configure API Keys</h2>

<pre>
# Copy the environment template
copy .env.example .env

# Edit .env file and add your API keys
notepad .env
</pre>

<h2>3.5 Verify Installation</h2>

<pre>
# Test that the application starts correctly
python main.py --help
</pre>

---

<h1>4. API CONFIGURATION</h1>

<h2>4.1 Google Cloud Vision API Setup</h2>

<p><b>Step 1: Create Google Cloud Account</b></p>
<p>Visit https://cloud.google.com and sign up. New users receive $300 in free credits.</p>

<p><b>Step 2: Create a New Project</b></p>
<p>Go to Google Cloud Console → Click project dropdown → New Project → Name it "ImageRecognition"</p>

<p><b>Step 3: Enable Vision API</b></p>
<p>Navigate to APIs & Services → Library → Search "Cloud Vision API" → Click Enable</p>

<p><b>Step 4: Create API Key</b></p>
<p>Go to APIs & Services → Credentials → Create Credentials → API Key → Copy the key</p>

<p><b>Step 5: Add to .env File</b></p>
<pre>
GOOGLE_VISION_API_KEY=your_api_key_here
</pre>

<h2>4.2 Roboflow API Setup</h2>

<p><b>Step 1: Create Roboflow Account</b></p>
<p>Visit https://roboflow.com and create a free account.</p>

<p><b>Step 2: Get API Key</b></p>
<p>Go to Settings → API → Copy your Private API Key</p>

<p><b>Step 3: Add to .env File</b></p>
<pre>
ROBOFLOW_API_KEY=your_roboflow_key_here
</pre>

<h2>4.3 Clarifai API Setup (Optional)</h2>

<p><b>Step 1: Create Clarifai Account</b></p>
<p>Visit https://clarifai.com and sign up for a free account.</p>

<p><b>Step 2: Get API Key</b></p>
<p>Go to Settings → Security → Create Personal Access Token</p>

<p><b>Step 3: Add to .env File</b></p>
<pre>
CLARIFAI_API_KEY=your_clarifai_key_here
</pre>

---

<h1>5. USAGE INSTRUCTIONS</h1>

<h2>5.1 Real-Time Camera Detection</h2>

<pre>
# Start real-time detection with default camera
python main.py --mode realtime

# Use specific camera (0=built-in, 1=external USB camera)
python main.py --mode realtime --camera 1
</pre>

<h2>5.2 Process Single Image</h2>

<pre>
# Analyze a single image file
python main.py --mode image --input path/to/your/image.jpg

# Results will be saved in the 'results' folder
</pre>

<h2>5.3 Process Video File</h2>

<pre>
# Analyze a video file
python main.py --mode video --input path/to/your/video.mp4

# Processed video will be saved in the 'results' folder
</pre>

<h2>5.4 Keyboard Controls (During Real-Time Mode)</h2>

<table>
<tr><th>Key</th><th>Action</th></tr>
<tr><td>Q</td><td>Quit the application</td></tr>
<tr><td>M</td><td>Toggle mode (Detection → Cursor → 3D)</td></tr>
<tr><td>C</td><td>Create cube (3D mode)</td></tr>
<tr><td>P</td><td>Create pyramid (3D mode)</td></tr>
<tr><td>S</td><td>Create sphere (3D mode)</td></tr>
<tr><td>Y</td><td>Create cylinder (3D mode)</td></tr>
<tr><td>X</td><td>Clear all 3D structures</td></tr>
</table>

---

<h1>6. GESTURE CONTROLS</h1>

<h2>6.1 Available Gestures (Cursor Mode)</h2>

<table>
<tr><th>Gesture</th><th>Description</th><th>Action</th></tr>
<tr><td>Pointing</td><td>Index finger extended, others closed</td><td>Move cursor</td></tr>
<tr><td>Pinch</td><td>Thumb and index finger touch</td><td>Left click</td></tr>
<tr><td>Double Pinch</td><td>Pinch twice quickly</td><td>Double click</td></tr>
<tr><td>Three-Finger Pinch</td><td>Thumb, index, and middle touch</td><td>Right click</td></tr>
<tr><td>Peace Sign</td><td>Index and middle fingers up</td><td>Scroll mode</td></tr>
<tr><td>Peace + Move Up/Down</td><td>Move peace sign vertically</td><td>Scroll up/down</td></tr>
<tr><td>Fist</td><td>All fingers closed</td><td>Drag mode</td></tr>
<tr><td>Open Palm</td><td>All fingers extended</td><td>Release/Stop</td></tr>
<tr><td>Thumbs Up</td><td>Thumb up, others closed</td><td>Confirm action</td></tr>
<tr><td>Thumbs Down</td><td>Thumb down, others closed</td><td>Cancel action</td></tr>
</table>

<h2>6.2 Gesture Tips</h2>

<p>
• Keep your hand 30-60 cm from the camera for best detection<br>
• Ensure good lighting for accurate gesture recognition<br>
• Move slowly and deliberately when performing gestures<br>
• The cursor uses Kalman filter smoothing for stability
</p>

---

<h1>7. API CHARACTERISTICS</h1>

<h2>7.1 Google Cloud Vision API</h2>

<table>
<tr><th>Feature</th><th>Details</th></tr>
<tr><td>Free Tier</td><td>1,000 units per month (1 unit = 1 feature per image)</td></tr>
<tr><td>New User Credits</td><td>$300 free credits for 90 days</td></tr>
<tr><td>Supported Features</td><td>Object Localization, Text Detection (OCR), Face Detection, Label Detection</td></tr>
<tr><td>Response Time</td><td>Typically 1-3 seconds per image</td></tr>
<tr><td>Image Size Limit</td><td>20 MB maximum file size</td></tr>
<tr><td>Rate Limit</td><td>1,800 requests per minute</td></tr>
<tr><td>Accuracy</td><td>High accuracy for common objects, excellent OCR for printed text</td></tr>
</table>

<h2>7.2 Roboflow API</h2>

<table>
<tr><th>Feature</th><th>Details</th></tr>
<tr><td>Free Tier</td><td>1,000 inference credits per month</td></tr>
<tr><td>Specialization</td><td>Custom trained models for specific objects (currency, medicine, etc.)</td></tr>
<tr><td>Response Time</td><td>Typically 0.5-2 seconds per image</td></tr>
<tr><td>Model Hosting</td><td>Host and deploy custom trained models</td></tr>
<tr><td>Accuracy</td><td>Varies by model; custom models can achieve 90%+ accuracy</td></tr>
</table>

<h2>7.3 Clarifai API (Optional)</h2>

<table>
<tr><th>Feature</th><th>Details</th></tr>
<tr><td>Free Tier</td><td>1,000 operations per month</td></tr>
<tr><td>Features</td><td>General image recognition, concept detection, custom training</td></tr>
<tr><td>Response Time</td><td>Typically 1-2 seconds per image</td></tr>
<tr><td>Pre-trained Models</td><td>Various models for food, apparel, travel, etc.</td></tr>
</table>

<h2>7.4 Local Detection (YOLOv8)</h2>

<table>
<tr><th>Feature</th><th>Details</th></tr>
<tr><td>Models Included</td><td>yolov8n.pt (fast), yolov8x.pt (accurate)</td></tr>
<tr><td>Object Classes</td><td>80 COCO dataset classes</td></tr>
<tr><td>Speed (GPU)</td><td>30-60 FPS with NVIDIA GPU</td></tr>
<tr><td>Speed (CPU)</td><td>5-15 FPS depending on processor</td></tr>
<tr><td>Offline Capable</td><td>Yes, works without internet connection</td></tr>
<tr><td>Accuracy</td><td>Good for common objects; cloud APIs may be more accurate for specialized detection</td></tr>
</table>

<h2>7.5 Hybrid Detection Strategy</h2>

<p>The system uses an intelligent hybrid approach:</p>

<p>
1. <b>Local First</b> - YOLOv8 processes frames locally for speed<br>
2. <b>Cloud Verification</b> - Important detections are verified with cloud APIs for accuracy<br>
3. <b>Automatic Fallback</b> - If cloud APIs are unavailable or quota exceeded, local detection continues<br>
4. <b>Result Caching</b> - Cloud results are cached for 1 second to reduce redundant API calls<br>
5. <b>Rate Limiting</b> - Built-in rate limiting to stay within API free tiers
</p>

---

<h1>8. PROJECT STRUCTURE</h1>

<pre>
Image Recognition/
├── main.py                     # Main application entry point
├── config.yaml                 # Configuration settings
├── requirements.txt            # Python dependencies
├── .env.example               # API key template
├── .env                       # Your API keys (create from .env.example)
├── .gitignore                 # Git ignore rules
│
├── detectors/                 # Detection modules
│   ├── __init__.py
│   ├── object_detector.py     # YOLOv8 + cloud object detection
│   ├── cloud_vision_detector.py # Unified cloud API wrapper
│   ├── currency_detector.py   # Currency note/coin detection
│   ├── hand_gesture_detector.py # Hand gesture recognition
│   ├── face_eye_tracker.py    # Face and eye detection
│   ├── ocr_engine.py          # Text recognition
│   ├── depth_estimator.py     # Distance estimation
│   ├── medicine_detector.py   # Medicine identification
│   └── food_detector.py       # Food recognition
│
├── utils/                     # Utility modules
│   ├── __init__.py
│   ├── api_key_manager.py     # Secure API key handling
│   ├── cursor_controller.py   # Mouse control with Kalman filter
│   ├── camera_manager.py      # Camera handling
│   ├── visualization.py       # Drawing and display
│   ├── result_aggregator.py   # Combine detection results
│   ├── performance_optimizer.py # GPU/CPU optimization
│   └── structure_3d_creator.py # 3D object creation
│
├── results/                   # Output directory for processed files
│
├── yolov8n.pt                # YOLOv8 Nano model (fast)
└── yolov8x.pt                # YOLOv8 Extra-large model (accurate)
</pre>

---

<h1>9. TROUBLESHOOTING</h1>

<h2>9.1 Common Issues</h2>

<table>
<tr><th>Problem</th><th>Solution</th></tr>
<tr><td>pip install fails</td><td>Use <code>py -m pip install -r requirements.txt</code></td></tr>
<tr><td>Camera not detected</td><td>Try different camera index: <code>--camera 1</code> or <code>--camera 2</code></td></tr>
<tr><td>CUDA out of memory</td><td>Use smaller model: change <code>yolov8x</code> to <code>yolov8n</code> in config.yaml</td></tr>
<tr><td>Cloud API not working</td><td>Check .env file has correct API keys; verify API is enabled in cloud console</td></tr>
<tr><td>Slow performance</td><td>Disable cloud APIs in config.yaml or use GPU acceleration</td></tr>
<tr><td>Hand gestures not detected</td><td>Improve lighting; keep hand 30-60cm from camera</td></tr>
</table>

<h2>9.2 Error Messages</h2>

<p><b>"No API key found for google_vision"</b></p>
<p>Create .env file from .env.example and add your Google Vision API key.</p>

<p><b>"Failed to load primary model"</b></p>
<p>Ensure yolov8x.pt or yolov8n.pt files exist in the project directory.</p>

<p><b>"Camera read failed"</b></p>
<p>Check if another application is using the camera. Try restarting the application.</p>

---

<h1>10. LICENSE</h1>

<p>This project is provided for educational and personal use. Commercial use may require additional licensing for the following components:</p>

<p>
• YOLOv8 - AGPL-3.0 License (Ultralytics)<br>
• MediaPipe - Apache 2.0 License (Google)<br>
• Google Cloud Vision - Subject to Google Cloud Terms of Service<br>
• Roboflow - Subject to Roboflow Terms of Service
</p>

---

<p style="text-align: center; margin-top: 30px;">
<b>Advanced Multi-Modal Image Recognition System</b><br>
Version 2.0 | February 2026<br>
Built with Python, YOLOv8, MediaPipe, and Cloud AI
</p>

</body>
</html>
