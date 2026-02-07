"""
Food detection and recognition
"""

import cv2
import numpy as np
import torch
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


class FoodDetector:
    """Detect and identify food items"""
    
    def __init__(self, config: Dict, optimizer):
        self.config = config
        self.optimizer = optimizer
        self.model = None
        
        self._load_model()
    
    def _load_model(self):
        """Load food detection model"""
        try:
            from ultralytics import YOLO
            
            # Try custom food model
            try:
                self.model = YOLO('models/food_detector.pt')
                logger.info("Custom food model loaded")
            except:
                # Fallback to general YOLOv8 (has many food classes)
                self.model = YOLO('yolov8x.pt')
                logger.info("Using YOLOv8 for food detection")
            
            if torch.cuda.is_available() and self.optimizer.use_gpu:
                self.model.to('cuda')
                
        except Exception as e:
            logger.warning(f"Food model loading failed: {e}")
            self.model = None
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect food items in frame
        Returns:
        [{
            'food_name': str,
            'category': str,
            'confidence': float,
            'bbox': [x1, y1, x2, y2]
        }]
        """
        if self.model is None:
            return []
        
        try:
            results = self.model(frame, conf=0.4, verbose=False)
            
            detections = []
            
            # Food-related classes in COCO dataset
            food_classes = [
                'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                'hot dog', 'pizza', 'donut', 'cake', 'bottle', 'wine glass',
                'cup', 'fork', 'knife', 'spoon', 'bowl'
            ]
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = self.model.names[cls]
                    
                    # Filter for food items
                    if class_name.lower() in food_classes or 'food' in class_name.lower():
                        detections.append({
                            'food_name': class_name,
                            'category': self._categorize_food(class_name),
                            'confidence': conf,
                            'bbox': [int(x1), int(y1), int(x2), int(y2)]
                        })
            
            return detections
            
        except Exception as e:
            logger.warning(f"Food detection failed: {e}")
            return []
    
    def _categorize_food(self, food_name: str) -> str:
        """Categorize food into groups"""
        categories = {
            'fruit': ['banana', 'apple', 'orange'],
            'vegetable': ['broccoli', 'carrot'],
            'fast_food': ['hot dog', 'pizza', 'sandwich'],
            'dessert': ['donut', 'cake'],
            'beverage': ['bottle', 'wine glass', 'cup'],
            'utensil': ['fork', 'knife', 'spoon', 'bowl']
        }
        
        for category, items in categories.items():
            if food_name.lower() in items:
                return category
        
        return 'other'
