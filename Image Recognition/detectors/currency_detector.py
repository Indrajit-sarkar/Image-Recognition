"""
Currency detection - Enhanced with Cloud APIs for better accuracy
Supports: Google Vision OCR, Roboflow custom models, and local fallback
"""

import cv2
import numpy as np
import torch
import logging
from typing import List, Dict, Optional
import re

logger = logging.getLogger(__name__)


class CurrencyDetector:
    """Detect and identify currency notes and coins with cloud API support"""
    
    # Known currency denominations
    CURRENCY_INFO = {
        'USD': {
            'symbol': '$',
            'notes': [1, 2, 5, 10, 20, 50, 100],
            'colors': {
                1: (200, 200, 200),
                5: (180, 180, 200),
                10: (200, 180, 150),
                20: (180, 200, 180),
                50: (200, 180, 200),
                100: (180, 200, 200),
            }
        },
        'EUR': {
            'symbol': '€',
            'notes': [5, 10, 20, 50, 100, 200, 500],
            'colors': {}
        },
        'GBP': {
            'symbol': '£',
            'notes': [5, 10, 20, 50],
            'colors': {}
        },
        'INR': {
            'symbol': '₹',
            'notes': [10, 20, 50, 100, 200, 500, 2000],
            'colors': {
                10: (180, 120, 80),    # Brown
                20: (120, 180, 120),   # Green
                50: (100, 200, 180),   # Teal
                100: (120, 120, 180),  # Blue
                200: (200, 150, 100),  # Orange
                500: (100, 100, 100),  # Gray
                2000: (180, 100, 150), # Pink/Magenta
            }
        },
        'JPY': {
            'symbol': '¥',
            'notes': [1000, 5000, 10000],
            'colors': {}
        },
        'CNY': {
            'symbol': '¥',
            'notes': [1, 5, 10, 20, 50, 100],
            'colors': {}
        },
        'AUD': {
            'symbol': '$',
            'notes': [5, 10, 20, 50, 100],
            'colors': {}
        },
        'CAD': {
            'symbol': '$',
            'notes': [5, 10, 20, 50, 100],
            'colors': {}
        }
    }
    
    def __init__(self, config: Dict, optimizer):
        self.config = config
        self.optimizer = optimizer
        self.model = None
        
        # Cloud detection
        self.use_cloud = config.get('use_cloud_detection', True)
        self.cloud_detector = None
        self._init_cloud_detector()
        
        # Local detection
        self._load_model()
        
        # Currency templates and features
        self.currency_database = self._load_currency_database()
    
    def _init_cloud_detector(self):
        """Initialize cloud vision detector for enhanced accuracy"""
        if self.use_cloud:
            try:
                from detectors.cloud_vision_detector import CloudVisionDetector
                cloud_config = self.config.get('cloud_apis', {})
                self.cloud_detector = CloudVisionDetector(cloud_config)
                
                if self.cloud_detector.is_available():
                    logger.info("Cloud detection enabled for currency")
                else:
                    logger.info("Cloud APIs not configured, using local detection only")
                    self.cloud_detector = None
                    
            except Exception as e:
                logger.warning(f"Cloud detector init failed: {e}")
                self.cloud_detector = None
    
    def _load_currency_database(self) -> Dict:
        """Load currency recognition database"""
        return self.CURRENCY_INFO
    
    def _load_model(self):
        """Load currency detection model"""
        try:
            from ultralytics import YOLO
            
            # Check if custom currency model exists
            try:
                self.model = YOLO('models/currency_detector.pt')
                logger.info("Custom currency model loaded")
            except:
                # Fallback to general object detection + feature matching
                self.model = YOLO('yolov8n.pt')
                logger.info("Using general model with feature matching for currency")
            
            if torch.cuda.is_available() and self.optimizer.use_gpu:
                self.model.to('cuda')
                
        except Exception as e:
            logger.warning(f"Currency model loading failed: {e}")
            self.model = None
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect currency in frame using best available method
        Returns:
        [{
            'type': 'note' or 'coin',
            'currency': 'USD', 'EUR', etc.,
            'denomination': int or float,
            'confidence': float,
            'bbox': [x1, y1, x2, y2],
            'source': 'cloud' or 'local'
        }]
        """
        detections = []
        
        # Method 1: Cloud API detection (highest accuracy)
        if self.cloud_detector:
            cloud_detections = self._detect_with_cloud(frame)
            if cloud_detections:
                detections.extend(cloud_detections)
        
        # Method 2: Local model-based detection
        if self.model and not detections:
            model_detections = self._detect_with_model(frame)
            detections.extend(model_detections)
        
        # Method 3: Feature-based detection (backup)
        if not detections:
            feature_detections = self._detect_with_features(frame)
            detections.extend(feature_detections)
        
        # Remove duplicates
        detections = self._remove_duplicates(detections)
        
        return detections
    
    def _detect_with_cloud(self, frame: np.ndarray) -> List[Dict]:
        """Detect currency using cloud APIs"""
        detections = []
        
        try:
            # Try Roboflow for specialized currency detection
            roboflow_results = self.cloud_detector.detect_currency_roboflow(frame)
            for result in roboflow_results:
                if result.get('type') != 'text_hint':
                    result['source'] = 'cloud_roboflow'
                    detections.append(result)
            
            if detections:
                return detections
            
            # Fall back to Google Vision OCR + object detection
            text_results = self.cloud_detector.detect_text_google(frame)
            object_results = self.cloud_detector.detect_objects_google(frame)
            
            # Analyze text for currency denominations
            currency_from_text = self._extract_currency_from_text(text_results)
            
            # Look for rectangular objects that could be notes
            for obj in object_results:
                obj_class = obj.get('class', '').lower()
                if any(word in obj_class for word in ['money', 'cash', 'note', 'bill', 'currency']):
                    detection = {
                        'type': 'note',
                        'currency': 'unknown',
                        'denomination': 0,
                        'confidence': obj.get('confidence', 0),
                        'bbox': obj.get('bbox', [0, 0, 0, 0]),
                        'source': 'cloud_google'
                    }
                    
                    # Try to match with text-detected denomination
                    if currency_from_text:
                        best_match = self._match_text_to_bbox(currency_from_text, detection['bbox'])
                        if best_match:
                            detection['currency'] = best_match.get('currency', 'unknown')
                            detection['denomination'] = best_match.get('denomination', 0)
                    
                    detections.append(detection)
            
            # If no money objects found but we have currency text, create detections from text
            if not detections and currency_from_text:
                for curr_text in currency_from_text:
                    detections.append({
                        'type': 'note',
                        'currency': curr_text.get('currency', 'unknown'),
                        'denomination': curr_text.get('denomination', 0),
                        'confidence': 0.7,
                        'bbox': curr_text.get('bbox', [0, 0, 0, 0]),
                        'source': 'cloud_ocr'
                    })
            
        except Exception as e:
            logger.warning(f"Cloud currency detection failed: {e}")
        
        return detections
    
    def _extract_currency_from_text(self, text_results: List[Dict]) -> List[Dict]:
        """Extract currency information from OCR text results"""
        currency_detections = []
        
        # Patterns for currency symbols and amounts
        patterns = [
            (r'\$(\d+)', 'USD'),
            (r'€(\d+)', 'EUR'),
            (r'£(\d+)', 'GBP'),
            (r'₹(\d+)', 'INR'),
            (r'¥(\d+)', 'JPY'),
            (r'(\d+)\s*dollars?', 'USD'),
            (r'(\d+)\s*euros?', 'EUR'),
            (r'(\d+)\s*pounds?', 'GBP'),
            (r'(\d+)\s*rupees?', 'INR'),
            (r'(\d+)\s*yen', 'JPY'),
        ]
        
        for text_item in text_results:
            text = text_item.get('text', '')
            bbox = text_item.get('bbox', [0, 0, 0, 0])
            
            for pattern, currency in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    try:
                        denomination = int(match.group(1))
                        if denomination in self.CURRENCY_INFO.get(currency, {}).get('notes', []):
                            currency_detections.append({
                                'currency': currency,
                                'denomination': denomination,
                                'bbox': bbox,
                                'text': text
                            })
                    except ValueError:
                        pass
            
            # Check for standalone numbers that match denominations
            if text.isdigit():
                value = int(text)
                for currency, info in self.CURRENCY_INFO.items():
                    if value in info.get('notes', []):
                        currency_detections.append({
                            'currency': currency,
                            'denomination': value,
                            'bbox': bbox,
                            'text': text
                        })
                        break
        
        return currency_detections
    
    def _match_text_to_bbox(self, text_items: List[Dict], target_bbox: List[int]) -> Optional[Dict]:
        """Find text item that best matches target bounding box"""
        if not text_items or not target_bbox:
            return None
        
        tx1, ty1, tx2, ty2 = target_bbox
        best_match = None
        best_overlap = 0
        
        for item in text_items:
            ix1, iy1, ix2, iy2 = item.get('bbox', [0, 0, 0, 0])
            
            # Calculate overlap
            overlap_x = max(0, min(tx2, ix2) - max(tx1, ix1))
            overlap_y = max(0, min(ty2, iy2) - max(ty1, iy1))
            overlap_area = overlap_x * overlap_y
            
            # Check if text is inside target bbox
            if ix1 >= tx1 and iy1 >= ty1 and ix2 <= tx2 and iy2 <= ty2:
                overlap_area *= 2  # Boost if fully contained
            
            if overlap_area > best_overlap:
                best_overlap = overlap_area
                best_match = item
        
        return best_match
    
    def _detect_with_model(self, frame: np.ndarray) -> List[Dict]:
        """Detect currency using trained model"""
        try:
            results = self.model(frame, conf=0.4, verbose=False)
            
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = self.model.names[cls]
                    
                    # Parse class name (e.g., "USD_20_note")
                    if 'note' in class_name.lower() or 'coin' in class_name.lower():
                        parts = class_name.split('_')
                        currency = parts[0] if len(parts) > 0 else 'unknown'
                        denomination = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
                        type_curr = 'note' if 'note' in class_name.lower() else 'coin'
                        
                        detections.append({
                            'type': type_curr,
                            'currency': currency,
                            'denomination': denomination,
                            'confidence': conf,
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'source': 'local_model'
                        })
            
            return detections
            
        except Exception as e:
            logger.warning(f"Model-based currency detection failed: {e}")
            return []
    
    def _detect_with_features(self, frame: np.ndarray) -> List[Dict]:
        """Detect currency using feature matching and color analysis"""
        detections = []
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect rectangular shapes (potential notes)
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Filter by area
                area = cv2.contourArea(contour)
                if area < 5000:  # Minimum area for currency
                    continue
                
                # Approximate contour
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                
                # Check if rectangular (4 corners)
                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(approx)
                    
                    # Check aspect ratio (typical for notes)
                    aspect_ratio = float(w) / h
                    if 1.5 < aspect_ratio < 3.0 or 0.33 < aspect_ratio < 0.67:
                        # Extract ROI
                        roi = frame[y:y+h, x:x+w]
                        
                        # Analyze color to identify currency
                        currency_info = self._identify_currency_by_color(roi)
                        
                        if currency_info:
                            detections.append({
                                'type': 'note',
                                'currency': currency_info['currency'],
                                'denomination': currency_info['denomination'],
                                'confidence': currency_info['confidence'],
                                'bbox': [x, y, x + w, y + h],
                                'source': 'local_features'
                            })
            
            # Detect circular shapes (potential coins)
            circles = cv2.HoughCircles(
                gray, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                param1=50, param2=30, minRadius=20, maxRadius=100
            )
            
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for circle in circles[0, :]:
                    cx, cy, r = circle
                    x1, y1 = max(0, cx - r), max(0, cy - r)
                    x2, y2 = min(frame.shape[1], cx + r), min(frame.shape[0], cy + r)
                    
                    detections.append({
                        'type': 'coin',
                        'currency': 'unknown',
                        'denomination': 0,
                        'confidence': 0.5,
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'source': 'local_features'
                    })
            
        except Exception as e:
            logger.warning(f"Feature-based currency detection failed: {e}")
        
        return detections
    
    def _identify_currency_by_color(self, roi: np.ndarray) -> Optional[Dict]:
        """Identify currency and denomination by color analysis"""
        if roi.size == 0:
            return None
        
        try:
            # Convert to HSV for better color analysis
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            avg_color = cv2.mean(roi)[:3]
            avg_hsv = cv2.mean(hsv)[:3]
            
            best_match = None
            best_distance = float('inf')
            
            # Compare with known currency colors
            for currency, info in self.CURRENCY_INFO.items():
                for denomination, expected_color in info.get('colors', {}).items():
                    # Calculate color distance
                    distance = np.sqrt(sum((a - b) ** 2 for a, b in zip(avg_color, expected_color)))
                    
                    if distance < best_distance and distance < 100:  # Threshold
                        best_distance = distance
                        best_match = {
                            'currency': currency,
                            'denomination': denomination,
                            'confidence': max(0.3, 1.0 - distance / 200)
                        }
            
            return best_match
            
        except Exception as e:
            logger.warning(f"Color analysis failed: {e}")
            return None
    
    def _remove_duplicates(self, detections: List[Dict]) -> List[Dict]:
        """Remove duplicate detections using NMS"""
        if len(detections) <= 1:
            return detections
        
        # Sort by confidence (prefer cloud > local)
        source_priority = {'cloud_roboflow': 3, 'cloud_google': 2, 'cloud_ocr': 2, 
                          'local_model': 1, 'local_features': 0}
        detections.sort(key=lambda x: (source_priority.get(x.get('source', ''), 0), 
                                       x.get('confidence', 0)), reverse=True)
        
        # Simple overlap-based deduplication
        filtered = []
        for det in detections:
            is_duplicate = False
            for existing in filtered:
                if self._iou(det['bbox'], existing['bbox']) > 0.5:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered.append(det)
        
        return filtered
    
    def _iou(self, box1: List, box2: List) -> float:
        """Calculate Intersection over Union"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
