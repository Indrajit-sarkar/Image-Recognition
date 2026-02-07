"""
Multi-language OCR with multiple engine fallbacks
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class OCREngine:
    """Multi-engine OCR with language detection"""
    
    def __init__(self, config: Dict, optimizer):
        self.config = config
        self.optimizer = optimizer
        self.engines = {}
        self.active_engine = None
        
        self._load_engines()
    
    def _load_engines(self):
        """Load OCR engines with fallback"""
        # Try EasyOCR (primary)
        try:
            import easyocr
            self.engines['easyocr'] = easyocr.Reader(
                self.config['languages'],
                gpu=self.optimizer.use_gpu
            )
            self.active_engine = 'easyocr'
            logger.info("EasyOCR loaded successfully")
        except Exception as e:
            logger.warning(f"EasyOCR failed: {e}")
        
        # Try PaddleOCR (backup)
        try:
            from paddleocr import PaddleOCR
            self.engines['paddleocr'] = PaddleOCR(
                use_angle_cls=True,
                lang='en',
                use_gpu=self.optimizer.use_gpu,
                show_log=False
            )
            if self.active_engine is None:
                self.active_engine = 'paddleocr'
            logger.info("PaddleOCR loaded successfully")
        except Exception as e:
            logger.warning(f"PaddleOCR failed: {e}")
        
        # Try Tesseract (backup)
        try:
            import pytesseract
            # Test if tesseract is available
            pytesseract.get_tesseract_version()
            self.engines['tesseract'] = pytesseract
            if self.active_engine is None:
                self.active_engine = 'tesseract'
            logger.info("Tesseract loaded successfully")
        except Exception as e:
            logger.warning(f"Tesseract failed: {e}")
    
    def detect_text(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect and recognize text in frame
        Returns list of text detections:
        [{
            'text': str,
            'confidence': float,
            'bbox': [x1, y1, x2, y2],
            'language': str (optional)
        }]
        """
        if not self.engines:
            # Return empty list silently when no OCR engine available
            return []
        
        try:
            if self.active_engine == 'easyocr':
                return self._detect_easyocr(frame)
            elif self.active_engine == 'paddleocr':
                return self._detect_paddleocr(frame)
            elif self.active_engine == 'tesseract':
                return self._detect_tesseract(frame)
        except Exception as e:
            logger.error(f"OCR failed with {self.active_engine}: {e}")
            return self._detect_with_fallback(frame)
        
        return []
    
    def _detect_easyocr(self, frame: np.ndarray) -> List[Dict]:
        """Detect text using EasyOCR"""
        reader = self.engines['easyocr']
        results = reader.readtext(frame)
        
        detections = []
        for bbox, text, conf in results:
            # Convert bbox format
            x_coords = [point[0] for point in bbox]
            y_coords = [point[1] for point in bbox]
            x1, y1 = int(min(x_coords)), int(min(y_coords))
            x2, y2 = int(max(x_coords)), int(max(y_coords))
            
            detections.append({
                'text': text,
                'confidence': float(conf),
                'bbox': [x1, y1, x2, y2],
                'language': 'auto'
            })
        
        return detections
    
    def _detect_paddleocr(self, frame: np.ndarray) -> List[Dict]:
        """Detect text using PaddleOCR"""
        ocr = self.engines['paddleocr']
        results = ocr.ocr(frame, cls=True)
        
        detections = []
        if results and results[0]:
            for line in results[0]:
                bbox_points = line[0]
                text = line[1][0]
                conf = line[1][1]
                
                # Convert bbox
                x_coords = [point[0] for point in bbox_points]
                y_coords = [point[1] for point in bbox_points]
                x1, y1 = int(min(x_coords)), int(min(y_coords))
                x2, y2 = int(max(x_coords)), int(max(y_coords))
                
                detections.append({
                    'text': text,
                    'confidence': float(conf),
                    'bbox': [x1, y1, x2, y2],
                    'language': 'auto'
                })
        
        return detections
    
    def _detect_tesseract(self, frame: np.ndarray) -> List[Dict]:
        """Detect text using Tesseract"""
        import pytesseract
        
        # Get detailed data
        data = pytesseract.image_to_data(frame, output_type=pytesseract.Output.DICT)
        
        detections = []
        n_boxes = len(data['text'])
        
        for i in range(n_boxes):
            if int(data['conf'][i]) > 0:
                text = data['text'][i].strip()
                if text:
                    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    detections.append({
                        'text': text,
                        'confidence': float(data['conf'][i]) / 100.0,
                        'bbox': [x, y, x + w, y + h],
                        'language': 'auto'
                    })
        
        return detections
    
    def _detect_with_fallback(self, frame: np.ndarray) -> List[Dict]:
        """Try OCR with fallback engines"""
        for engine_name in ['paddleocr', 'tesseract', 'easyocr']:
            if engine_name in self.engines and engine_name != self.active_engine:
                try:
                    logger.info(f"Trying fallback OCR: {engine_name}")
                    self.active_engine = engine_name
                    return self.detect_text(frame)
                except Exception as e:
                    logger.warning(f"Fallback {engine_name} failed: {e}")
        
        return []
