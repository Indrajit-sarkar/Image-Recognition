"""
Camera management and frame capture
"""

import cv2
import numpy as np
import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)


class CameraManager:
    """Manage camera input with optimization"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.cap = None
        self.source = config.get('default_source', 0)
        
        self._initialize_camera()
    
    def _initialize_camera(self):
        """Initialize camera with optimal settings"""
        try:
            self.cap = cv2.VideoCapture(self.source)
            
            if not self.cap.isOpened():
                raise Exception(f"Failed to open camera {self.source}")
            
            # Set resolution
            width, height = self.config.get('resolution', [1920, 1080])
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            # Set FPS
            fps = self.config.get('fps', 30)
            self.cap.set(cv2.CAP_PROP_FPS, fps)
            
            # Set buffer size (lower = less latency)
            buffer_size = self.config.get('buffer_size', 1)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
            
            # Enable auto-focus and auto-exposure
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            
            logger.info(f"Camera {self.source} initialized: {width}x{height} @ {fps}fps")
            
        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
            raise
    
    def read_frame(self) -> Optional[np.ndarray]:
        """Read a frame from camera"""
        if self.cap is None or not self.cap.isOpened():
            return None
        
        ret, frame = self.cap.read()
        
        if not ret:
            logger.warning("Failed to read frame")
            return None
        
        return frame
    
    def set_source(self, source: int):
        """Change camera source"""
        if source != self.source:
            self.release()
            self.source = source
            self._initialize_camera()
    
    def release(self):
        """Release camera resources"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            logger.info("Camera released")
    
    def __del__(self):
        """Cleanup on deletion"""
        self.release()
