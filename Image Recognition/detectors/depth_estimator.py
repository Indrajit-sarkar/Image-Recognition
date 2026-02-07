"""
Depth estimation for distance calculation
"""

import cv2
import numpy as np
import torch
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class DepthEstimator:
    """Estimate depth and distance from monocular camera"""
    
    def __init__(self, config: Dict, optimizer):
        self.config = config
        self.optimizer = optimizer
        self.model = None
        self.active_model = None
        
        self._load_model()
    
    def _load_model(self):
        """Load depth estimation model"""
        try:
            # Try Depth-Anything (state-of-the-art)
            self._load_depth_anything()
        except Exception as e:
            logger.warning(f"Depth-Anything failed: {e}")
            try:
                # Fallback to MiDaS
                self._load_midas()
            except Exception as e2:
                logger.error(f"All depth models failed: {e2}")
    
    def _load_depth_anything(self):
        """Load Depth-Anything model"""
        try:
            from transformers import pipeline
            
            self.model = pipeline(
                task="depth-estimation",
                model="LiheYoung/depth-anything-small-hf"
            )
            
            if torch.cuda.is_available() and self.optimizer.use_gpu:
                self.model.model.to('cuda')
            
            self.active_model = 'depth_anything'
            logger.info("Depth-Anything loaded successfully")
            
        except Exception as e:
            raise Exception(f"Depth-Anything loading failed: {e}")
    
    def _load_midas(self):
        """Load MiDaS model as fallback"""
        try:
            self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
            self.model.eval()
            
            if torch.cuda.is_available() and self.optimizer.use_gpu:
                self.model.to('cuda')
            
            # Load transforms
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            self.transform = midas_transforms.small_transform
            
            self.active_model = 'midas'
            logger.info("MiDaS loaded successfully")
            
        except Exception as e:
            raise Exception(f"MiDaS loading failed: {e}")
    
    def estimate(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Estimate depth map from frame
        Returns depth map as numpy array (normalized 0-1)
        """
        if self.model is None:
            return None
        
        try:
            if self.active_model == 'depth_anything':
                return self._estimate_depth_anything(frame)
            elif self.active_model == 'midas':
                return self._estimate_midas(frame)
        except Exception as e:
            logger.error(f"Depth estimation failed: {e}")
            return None
    
    def _estimate_depth_anything(self, frame: np.ndarray) -> np.ndarray:
        """Estimate depth using Depth-Anything"""
        from PIL import Image
        
        # Convert to PIL Image
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Predict
        result = self.model(image)
        depth = np.array(result['depth'])
        
        # Normalize
        depth = (depth - depth.min()) / (depth.max() - depth.min())
        
        # Resize to match input
        depth = cv2.resize(depth, (frame.shape[1], frame.shape[0]))
        
        return depth
    
    def _estimate_midas(self, frame: np.ndarray) -> np.ndarray:
        """Estimate depth using MiDaS"""
        # Prepare input
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img)
        
        if torch.cuda.is_available() and self.optimizer.use_gpu:
            input_batch = input_batch.to('cuda')
        
        # Predict
        with torch.no_grad():
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        depth = prediction.cpu().numpy()
        
        # Normalize
        depth = (depth - depth.min()) / (depth.max() - depth.min())
        
        return depth
    
    def get_distance(self, depth_map: np.ndarray, bbox: list) -> float:
        """
        Calculate distance to object from depth map
        bbox: [x1, y1, x2, y2]
        Returns distance in meters (estimated)
        """
        if depth_map is None:
            return -1.0
        
        x1, y1, x2, y2 = bbox
        
        # Get depth in bounding box region
        roi_depth = depth_map[y1:y2, x1:x2]
        
        if roi_depth.size == 0:
            return -1.0
        
        # Use median depth of object
        median_depth = np.median(roi_depth)
        
        # Convert normalized depth to approximate distance
        # This is a rough estimation - calibration needed for accuracy
        max_distance = self.config.get('max_distance', 10.0)
        distance = (1.0 - median_depth) * max_distance
        
        return float(distance)
