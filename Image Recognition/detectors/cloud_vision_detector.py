"""
Cloud Vision Detector - Unified cloud API wrapper for enhanced detection
Supports: Google Cloud Vision, Roboflow, Clarifai
"""

import cv2
import numpy as np
import base64
import logging
import time
import requests
from typing import List, Dict, Optional, Any
from functools import lru_cache
import hashlib

logger = logging.getLogger(__name__)


class CloudVisionDetector:
    """Multi-API cloud vision detection for enhanced accuracy"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.enabled = config.get('enabled', True)
        
        # Rate limiting
        self.max_requests_per_minute = config.get('max_requests_per_minute', 60)
        self.request_times: List[float] = []
        
        # Caching
        self.cache_enabled = config.get('cache_enabled', True)
        self.cache_ttl = config.get('cache_ttl', 1.0)  # seconds
        self._cache: Dict[str, tuple] = {}  # hash -> (result, timestamp)
        
        # Initialize API clients
        self._init_google_vision()
        self._init_roboflow()
        self._init_clarifai()
        
        # Track API availability
        self.available_apis = self._check_available_apis()
        logger.info(f"Cloud Vision initialized. Available APIs: {self.available_apis}")
    
    def _init_google_vision(self):
        """Initialize Google Cloud Vision API"""
        self.google_enabled = False
        try:
            from utils.api_key_manager import APIKeyManager
            self.google_api_key = APIKeyManager.get_key('google_vision')
            
            if self.google_api_key:
                self.google_enabled = True
                self.google_endpoint = "https://vision.googleapis.com/v1/images:annotate"
                logger.info("Google Cloud Vision API initialized")
        except Exception as e:
            logger.warning(f"Google Vision init failed: {e}")
    
    def _init_roboflow(self):
        """Initialize Roboflow API"""
        self.roboflow_enabled = False
        try:
            from utils.api_key_manager import APIKeyManager
            self.roboflow_api_key = APIKeyManager.get_key('roboflow')
            
            if self.roboflow_api_key:
                self.roboflow_enabled = True
                # Default currency detection model - can be configured
                self.roboflow_model = self.config.get('roboflow', {}).get(
                    'currency_model', 'currency-detection'
                )
                logger.info("Roboflow API initialized")
        except Exception as e:
            logger.warning(f"Roboflow init failed: {e}")
    
    def _init_clarifai(self):
        """Initialize Clarifai API"""
        self.clarifai_enabled = False
        try:
            from utils.api_key_manager import APIKeyManager
            self.clarifai_api_key = APIKeyManager.get_key('clarifai')
            
            if self.clarifai_api_key:
                self.clarifai_enabled = True
                logger.info("Clarifai API initialized")
        except Exception as e:
            logger.warning(f"Clarifai init failed: {e}")
    
    def _check_available_apis(self) -> List[str]:
        """Check which APIs are available"""
        available = []
        if self.google_enabled:
            available.append('google_vision')
        if self.roboflow_enabled:
            available.append('roboflow')
        if self.clarifai_enabled:
            available.append('clarifai')
        return available
    
    def _rate_limit_check(self) -> bool:
        """Check if we're within rate limits"""
        current_time = time.time()
        # Remove requests older than 1 minute
        self.request_times = [t for t in self.request_times if current_time - t < 60]
        
        if len(self.request_times) >= self.max_requests_per_minute:
            logger.warning("Rate limit reached, using local detection")
            return False
        
        self.request_times.append(current_time)
        return True
    
    def _get_cache_key(self, frame: np.ndarray, feature: str) -> str:
        """Generate cache key from frame and feature"""
        # Use reduced frame for faster hashing
        small_frame = cv2.resize(frame, (64, 64))
        frame_hash = hashlib.md5(small_frame.tobytes()).hexdigest()
        return f"{feature}_{frame_hash}"
    
    def _check_cache(self, key: str) -> Optional[Any]:
        """Check if result is in cache and still valid"""
        if not self.cache_enabled:
            return None
        
        if key in self._cache:
            result, timestamp = self._cache[key]
            if time.time() - timestamp < self.cache_ttl:
                return result
            else:
                del self._cache[key]
        return None
    
    def _set_cache(self, key: str, result: Any):
        """Store result in cache"""
        if self.cache_enabled:
            self._cache[key] = (result, time.time())
    
    def _frame_to_base64(self, frame: np.ndarray) -> str:
        """Convert frame to base64 for API requests"""
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buffer).decode('utf-8')
    
    # ========== Google Cloud Vision API Methods ==========
    
    def detect_objects_google(self, frame: np.ndarray) -> List[Dict]:
        """Detect objects using Google Cloud Vision API"""
        if not self.google_enabled:
            return []
        
        cache_key = self._get_cache_key(frame, 'google_objects')
        cached = self._check_cache(cache_key)
        if cached is not None:
            return cached
        
        if not self._rate_limit_check():
            return []
        
        try:
            image_content = self._frame_to_base64(frame)
            
            request_body = {
                "requests": [{
                    "image": {"content": image_content},
                    "features": [
                        {"type": "OBJECT_LOCALIZATION", "maxResults": 20},
                        {"type": "LABEL_DETECTION", "maxResults": 10}
                    ]
                }]
            }
            
            response = requests.post(
                f"{self.google_endpoint}?key={self.google_api_key}",
                json=request_body,
                timeout=5
            )
            
            if response.status_code != 200:
                logger.warning(f"Google Vision API error: {response.status_code}")
                return []
            
            data = response.json()
            detections = self._parse_google_objects(data, frame.shape)
            
            self._set_cache(cache_key, detections)
            return detections
            
        except Exception as e:
            logger.error(f"Google Vision object detection failed: {e}")
            return []
    
    def _parse_google_objects(self, data: Dict, frame_shape: tuple) -> List[Dict]:
        """Parse Google Vision object localization response"""
        detections = []
        h, w = frame_shape[:2]
        
        try:
            responses = data.get('responses', [])
            if not responses:
                return []
            
            # Parse localized objects
            objects = responses[0].get('localizedObjectAnnotations', [])
            for obj in objects:
                vertices = obj.get('boundingPoly', {}).get('normalizedVertices', [])
                if len(vertices) >= 4:
                    x1 = int(vertices[0].get('x', 0) * w)
                    y1 = int(vertices[0].get('y', 0) * h)
                    x2 = int(vertices[2].get('x', 0) * w)
                    y2 = int(vertices[2].get('y', 0) * h)
                    
                    detections.append({
                        'class': obj.get('name', 'unknown'),
                        'confidence': obj.get('score', 0),
                        'bbox': [x1, y1, x2, y2],
                        'center': [(x1 + x2) // 2, (y1 + y2) // 2],
                        'source': 'google_vision'
                    })
            
        except Exception as e:
            logger.error(f"Failed to parse Google Vision response: {e}")
        
        return detections
    
    def detect_text_google(self, frame: np.ndarray) -> List[Dict]:
        """Detect text using Google Cloud Vision OCR"""
        if not self.google_enabled:
            return []
        
        cache_key = self._get_cache_key(frame, 'google_text')
        cached = self._check_cache(cache_key)
        if cached is not None:
            return cached
        
        if not self._rate_limit_check():
            return []
        
        try:
            image_content = self._frame_to_base64(frame)
            
            request_body = {
                "requests": [{
                    "image": {"content": image_content},
                    "features": [
                        {"type": "TEXT_DETECTION", "maxResults": 20}
                    ]
                }]
            }
            
            response = requests.post(
                f"{self.google_endpoint}?key={self.google_api_key}",
                json=request_body,
                timeout=5
            )
            
            if response.status_code != 200:
                return []
            
            data = response.json()
            detections = self._parse_google_text(data, frame.shape)
            
            self._set_cache(cache_key, detections)
            return detections
            
        except Exception as e:
            logger.error(f"Google Vision OCR failed: {e}")
            return []
    
    def _parse_google_text(self, data: Dict, frame_shape: tuple) -> List[Dict]:
        """Parse Google Vision text detection response"""
        detections = []
        h, w = frame_shape[:2]
        
        try:
            responses = data.get('responses', [])
            if not responses:
                return []
            
            annotations = responses[0].get('textAnnotations', [])
            # First annotation is full text, skip it for individual words
            for annotation in annotations[1:]:
                text = annotation.get('description', '')
                vertices = annotation.get('boundingPoly', {}).get('vertices', [])
                
                if len(vertices) >= 4:
                    x_coords = [v.get('x', 0) for v in vertices]
                    y_coords = [v.get('y', 0) for v in vertices]
                    
                    detections.append({
                        'text': text,
                        'bbox': [min(x_coords), min(y_coords), 
                                max(x_coords), max(y_coords)],
                        'source': 'google_vision'
                    })
            
        except Exception as e:
            logger.error(f"Failed to parse Google text response: {e}")
        
        return detections
    
    def detect_faces_google(self, frame: np.ndarray) -> List[Dict]:
        """Detect faces using Google Cloud Vision API"""
        if not self.google_enabled:
            return []
        
        cache_key = self._get_cache_key(frame, 'google_faces')
        cached = self._check_cache(cache_key)
        if cached is not None:
            return cached
        
        if not self._rate_limit_check():
            return []
        
        try:
            image_content = self._frame_to_base64(frame)
            
            request_body = {
                "requests": [{
                    "image": {"content": image_content},
                    "features": [
                        {"type": "FACE_DETECTION", "maxResults": 10}
                    ]
                }]
            }
            
            response = requests.post(
                f"{self.google_endpoint}?key={self.google_api_key}",
                json=request_body,
                timeout=5
            )
            
            if response.status_code != 200:
                return []
            
            data = response.json()
            detections = self._parse_google_faces(data, frame.shape)
            
            self._set_cache(cache_key, detections)
            return detections
            
        except Exception as e:
            logger.error(f"Google Vision face detection failed: {e}")
            return []
    
    def _parse_google_faces(self, data: Dict, frame_shape: tuple) -> List[Dict]:
        """Parse Google Vision face detection response"""
        detections = []
        
        try:
            responses = data.get('responses', [])
            if not responses:
                return []
            
            faces = responses[0].get('faceAnnotations', [])
            for face in faces:
                vertices = face.get('boundingPoly', {}).get('vertices', [])
                if len(vertices) >= 4:
                    x_coords = [v.get('x', 0) for v in vertices]
                    y_coords = [v.get('y', 0) for v in vertices]
                    
                    detections.append({
                        'bbox': [min(x_coords), min(y_coords),
                                max(x_coords), max(y_coords)],
                        'confidence': face.get('detectionConfidence', 0),
                        'joy': face.get('joyLikelihood', 'UNKNOWN'),
                        'anger': face.get('angerLikelihood', 'UNKNOWN'),
                        'source': 'google_vision'
                    })
            
        except Exception as e:
            logger.error(f"Failed to parse Google faces response: {e}")
        
        return detections
    
    # ========== Roboflow API Methods ==========
    
    def detect_currency_roboflow(self, frame: np.ndarray, 
                                  model_id: str = None) -> List[Dict]:
        """Detect currency using Roboflow trained model"""
        if not self.roboflow_enabled:
            return []
        
        cache_key = self._get_cache_key(frame, 'roboflow_currency')
        cached = self._check_cache(cache_key)
        if cached is not None:
            return cached
        
        if not self._rate_limit_check():
            return []
        
        try:
            model = model_id or self.roboflow_model
            
            # Encode image
            _, buffer = cv2.imencode('.jpg', frame)
            
            # Roboflow inference API
            response = requests.post(
                f"https://detect.roboflow.com/{model}",
                params={"api_key": self.roboflow_api_key},
                files={"file": buffer.tobytes()},
                timeout=10
            )
            
            if response.status_code != 200:
                logger.warning(f"Roboflow API error: {response.status_code}")
                return []
            
            data = response.json()
            detections = self._parse_roboflow_currency(data)
            
            self._set_cache(cache_key, detections)
            return detections
            
        except Exception as e:
            logger.error(f"Roboflow currency detection failed: {e}")
            return []
    
    def _parse_roboflow_currency(self, data: Dict) -> List[Dict]:
        """Parse Roboflow detection response for currency"""
        detections = []
        
        try:
            predictions = data.get('predictions', [])
            for pred in predictions:
                x = pred.get('x', 0)
                y = pred.get('y', 0)
                w = pred.get('width', 0)
                h = pred.get('height', 0)
                
                x1 = int(x - w / 2)
                y1 = int(y - h / 2)
                x2 = int(x + w / 2)
                y2 = int(y + h / 2)
                
                class_name = pred.get('class', 'unknown')
                
                # Parse currency info from class name (e.g., "USD_20", "INR_500")
                currency_type = 'unknown'
                denomination = 0
                
                if '_' in class_name:
                    parts = class_name.split('_')
                    currency_type = parts[0]
                    if len(parts) > 1 and parts[1].isdigit():
                        denomination = int(parts[1])
                
                detections.append({
                    'type': 'note',
                    'currency': currency_type,
                    'denomination': denomination,
                    'confidence': pred.get('confidence', 0),
                    'bbox': [x1, y1, x2, y2],
                    'source': 'roboflow'
                })
            
        except Exception as e:
            logger.error(f"Failed to parse Roboflow response: {e}")
        
        return detections
    
    # ========== Unified Detection Methods ==========
    
    def detect_objects(self, frame: np.ndarray) -> List[Dict]:
        """Detect objects using best available cloud API"""
        if not self.enabled or not self.available_apis:
            return []
        
        # Try Google Vision first (most comprehensive)
        if self.google_enabled:
            results = self.detect_objects_google(frame)
            if results:
                return results
        
        return []
    
    def detect_currency(self, frame: np.ndarray) -> List[Dict]:
        """Detect currency using best available cloud API"""
        if not self.enabled or not self.available_apis:
            return []
        
        # Try Roboflow first (specialized model)
        if self.roboflow_enabled:
            results = self.detect_currency_roboflow(frame)
            if results:
                return results
        
        # Fall back to Google Vision for text (denomination)
        if self.google_enabled:
            text_results = self.detect_text_google(frame)
            # Look for currency-related text
            currency_hints = []
            for text in text_results:
                content = text.get('text', '').upper()
                # Check for currency denominations
                if any(c in content for c in ['$', '€', '£', '₹', '¥']):
                    currency_hints.append(text)
                elif content.isdigit() and int(content) in [1, 2, 5, 10, 20, 50, 100, 200, 500, 2000]:
                    currency_hints.append(text)
            
            if currency_hints:
                return [{'type': 'text_hint', 'hints': currency_hints, 'source': 'google_ocr'}]
        
        return []
    
    def detect_people(self, frame: np.ndarray) -> List[Dict]:
        """Detect people using cloud APIs"""
        if not self.enabled or not self.available_apis:
            return []
        
        objects = self.detect_objects(frame)
        people = [obj for obj in objects if obj.get('class', '').lower() == 'person']
        return people
    
    def is_available(self) -> bool:
        """Check if any cloud API is available"""
        return len(self.available_apis) > 0
