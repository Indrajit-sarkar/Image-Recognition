"""
Medicine detection and identification
"""

import cv2
import numpy as np
import torch
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


class MedicineDetector:
    """Detect and identify medicines (pills, bottles, labels)"""
    
    def __init__(self, config: Dict, optimizer):
        self.config = config
        self.optimizer = optimizer
        self.model = None
        self.ocr_engine = None
        
        self._load_model()
        self._load_ocr()
    
    def _load_model(self):
        """Load medicine detection model"""
        try:
            from ultralytics import YOLO
            
            # Try custom medicine model
            try:
                self.model = YOLO('models/medicine_detector.pt')
                logger.info("Custom medicine model loaded")
            except:
                # Fallback to general detection
                self.model = YOLO('yolov8n.pt')
                logger.info("Using general model for medicine detection")
            
            if torch.cuda.is_available() and self.optimizer.use_gpu:
                self.model.to('cuda')
                
        except Exception as e:
            logger.warning(f"Medicine model loading failed: {e}")
    
    def _load_ocr(self):
        """Load OCR for reading medicine labels"""
        try:
            import easyocr
            self.ocr_engine = easyocr.Reader(['en'], gpu=self.optimizer.use_gpu)
            logger.info("OCR for medicine labels loaded")
        except Exception as e:
            logger.warning(f"OCR loading failed: {e}")
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect medicines in frame
        Returns:
        [{
            'type': 'pill', 'bottle', 'box', etc.,
            'name': str (if readable),
            'confidence': float,
            'bbox': [x1, y1, x2, y2],
            'label_text': str (if available)
        }]
        """
        detections = []
        
        # Detect medicine objects
        if self.model:
            object_detections = self._detect_objects(frame)
            detections.extend(object_detections)
        
        # Read labels with OCR
        if self.ocr_engine and detections:
            detections = self._read_labels(frame, detections)
        
        return detections
    
    def _detect_objects(self, frame: np.ndarray) -> List[Dict]:
        """Detect medicine objects"""
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
                    
                    # Filter for medicine-related objects
                    medicine_keywords = ['bottle', 'pill', 'tablet', 'capsule', 
                                       'medicine', 'drug', 'box', 'package']
                    
                    if any(keyword in class_name.lower() for keyword in medicine_keywords):
                        detections.append({
                            'type': class_name,
                            'name': 'unknown',
                            'confidence': conf,
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'label_text': ''
                        })
            
            return detections
            
        except Exception as e:
            logger.warning(f"Medicine object detection failed: {e}")
            return []
    
    def _read_labels(self, frame: np.ndarray, detections: List[Dict]) -> List[Dict]:
        """Read text from medicine labels"""
        for detection in detections:
            try:
                x1, y1, x2, y2 = detection['bbox']
                
                # Expand ROI slightly to capture full label
                padding = 10
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(frame.shape[1], x2 + padding)
                y2 = min(frame.shape[0], y2 + padding)
                
                roi = frame[y1:y2, x1:x2]
                
                if roi.size > 0:
                    # Read text
                    results = self.ocr_engine.readtext(roi)
                    
                    if results:
                        # Combine all text
                        texts = [text for (_, text, _) in results]
                        detection['label_text'] = ' '.join(texts)
                        
                        # Try to extract medicine name (usually first/largest text)
                        if texts:
                            detection['name'] = texts[0]
                            
            except Exception as e:
                logger.warning(f"Label reading failed: {e}")
        
        return detections
