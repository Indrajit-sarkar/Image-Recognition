"""
Face detection and eye tracking
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class FaceEyeTracker:
    """Face and eye detection/tracking"""
    
    def __init__(self, face_config: Dict, eye_config: Dict, optimizer):
        self.face_config = face_config
        self.eye_config = eye_config
        self.optimizer = optimizer
        self.face_detector = None
        self.eye_detector = None
        
        self._load_detectors()
    
    def _load_detectors(self):
        """Load face and eye detection models"""
        # Try MediaPipe (primary)
        try:
            import mediapipe as mp
            self.mp_face = mp.solutions.face_detection
            self.mp_face_mesh = mp.solutions.face_mesh
            
            self.face_detector = self.mp_face.FaceDetection(
                model_selection=1,
                min_detection_confidence=0.5
            )
            self.eye_detector = self.mp_face_mesh.FaceMesh(
                max_num_faces=5,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            logger.info("MediaPipe face/eye detection loaded")
            self.active_detector = 'mediapipe'
            
        except Exception as e:
            logger.warning(f"MediaPipe failed: {e}")
            self._load_dlib_fallback()
    
    def _load_dlib_fallback(self):
        """Load dlib as fallback"""
        try:
            import dlib
            
            # Load face detector
            self.face_detector = dlib.get_frontal_face_detector()
            
            # Load shape predictor for eye detection
            # Note: Requires shape_predictor_68_face_landmarks.dat
            try:
                self.eye_detector = dlib.shape_predictor(
                    "models/shape_predictor_68_face_landmarks.dat"
                )
            except:
                logger.warning("Dlib shape predictor not found")
                self.eye_detector = None
            
            self.active_detector = 'dlib'
            logger.info("Dlib face detection loaded")
            
        except Exception as e:
            logger.error(f"Dlib fallback failed: {e}")
            self._load_opencv_fallback()
    
    def _load_opencv_fallback(self):
        """Load OpenCV Haar cascades as last resort"""
        try:
            self.face_detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            self.eye_detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_eye.xml'
            )
            self.active_detector = 'opencv'
            logger.info("OpenCV Haar cascades loaded")
            
        except Exception as e:
            logger.error(f"All face detectors failed: {e}")
    
    def detect(self, frame: np.ndarray) -> Dict:
        """
        Detect faces and eyes in frame
        Returns:
        {
            'faces': [{
                'bbox': [x1, y1, x2, y2],
                'confidence': float,
                'landmarks': {...}
            }],
            'eyes': [{
                'bbox': [x1, y1, x2, y2],
                'gaze_direction': [x, y] (optional)
            }]
        }
        """
        if self.face_detector is None:
            return {'faces': [], 'eyes': []}
        
        try:
            if self.active_detector == 'mediapipe':
                return self._detect_mediapipe(frame)
            elif self.active_detector == 'dlib':
                return self._detect_dlib(frame)
            elif self.active_detector == 'opencv':
                return self._detect_opencv(frame)
        except Exception as e:
            logger.error(f"Face/eye detection failed: {e}")
            return {'faces': [], 'eyes': []}
    
    def _detect_mediapipe(self, frame: np.ndarray) -> Dict:
        """Detect using MediaPipe"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        
        # Detect faces
        face_results = self.face_detector.process(rgb_frame)
        faces = []
        
        if face_results.detections:
            for detection in face_results.detections:
                bbox = detection.location_data.relative_bounding_box
                x1 = int(bbox.xmin * w)
                y1 = int(bbox.ymin * h)
                x2 = int((bbox.xmin + bbox.width) * w)
                y2 = int((bbox.ymin + bbox.height) * h)
                
                faces.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': detection.score[0],
                    'landmarks': {}
                })
        
        # Detect eyes with face mesh
        eyes = []
        mesh_results = self.eye_detector.process(rgb_frame)
        
        if mesh_results.multi_face_landmarks:
            for face_landmarks in mesh_results.multi_face_landmarks:
                # Left eye indices: 33, 133, 160, 159, 158, 157, 173
                # Right eye indices: 362, 263, 387, 386, 385, 384, 398
                
                # Left eye
                left_eye_points = [33, 133, 160, 159, 158, 157, 173]
                left_coords = [(int(face_landmarks.landmark[i].x * w),
                               int(face_landmarks.landmark[i].y * h))
                              for i in left_eye_points]
                
                if left_coords:
                    x_coords = [p[0] for p in left_coords]
                    y_coords = [p[1] for p in left_coords]
                    eyes.append({
                        'bbox': [min(x_coords), min(y_coords), max(x_coords), max(y_coords)],
                        'side': 'left'
                    })
                
                # Right eye
                right_eye_points = [362, 263, 387, 386, 385, 384, 398]
                right_coords = [(int(face_landmarks.landmark[i].x * w),
                                int(face_landmarks.landmark[i].y * h))
                               for i in right_eye_points]
                
                if right_coords:
                    x_coords = [p[0] for p in right_coords]
                    y_coords = [p[1] for p in right_coords]
                    eyes.append({
                        'bbox': [min(x_coords), min(y_coords), max(x_coords), max(y_coords)],
                        'side': 'right'
                    })
        
        return {'faces': faces, 'eyes': eyes}
    
    def _detect_dlib(self, frame: np.ndarray) -> Dict:
        """Detect using dlib"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        face_rects = self.face_detector(gray, 1)
        faces = []
        eyes = []
        
        for rect in face_rects:
            x1, y1 = rect.left(), rect.top()
            x2, y2 = rect.right(), rect.bottom()
            
            faces.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': 1.0,
                'landmarks': {}
            })
            
            # Detect eyes if shape predictor available
            if self.eye_detector:
                shape = self.eye_detector(gray, rect)
                
                # Left eye (points 36-41)
                left_eye = [(shape.part(i).x, shape.part(i).y) for i in range(36, 42)]
                x_coords = [p[0] for p in left_eye]
                y_coords = [p[1] for p in left_eye]
                eyes.append({
                    'bbox': [min(x_coords), min(y_coords), max(x_coords), max(y_coords)],
                    'side': 'left'
                })
                
                # Right eye (points 42-47)
                right_eye = [(shape.part(i).x, shape.part(i).y) for i in range(42, 48)]
                x_coords = [p[0] for p in right_eye]
                y_coords = [p[1] for p in right_eye]
                eyes.append({
                    'bbox': [min(x_coords), min(y_coords), max(x_coords), max(y_coords)],
                    'side': 'right'
                })
        
        return {'faces': faces, 'eyes': eyes}
    
    def _detect_opencv(self, frame: np.ndarray) -> Dict:
        """Detect using OpenCV Haar cascades"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        face_rects = self.face_detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        faces = []
        eyes = []
        
        for (x, y, w, h) in face_rects:
            faces.append({
                'bbox': [x, y, x + w, y + h],
                'confidence': 1.0,
                'landmarks': {}
            })
            
            # Detect eyes in face region
            roi_gray = gray[y:y+h, x:x+w]
            eye_rects = self.eye_detector.detectMultiScale(
                roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)
            )
            
            for (ex, ey, ew, eh) in eye_rects:
                eyes.append({
                    'bbox': [x + ex, y + ey, x + ex + ew, y + ey + eh],
                    'side': 'unknown'
                })
        
        return {'faces': faces, 'eyes': eyes}
