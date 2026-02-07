"""
Performance optimization utilities
"""

import torch
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class PerformanceOptimizer:
    """Optimize performance for detection models"""
    
    def __init__(self, config: Dict):
        self.config = config['performance']
        self.use_gpu = self.config.get('use_gpu', True) and torch.cuda.is_available()
        
        self._setup_optimization()
    
    def _setup_optimization(self):
        """Setup performance optimizations"""
        if self.use_gpu:
            # Set GPU memory fraction
            memory_fraction = self.config.get('gpu_memory_fraction', 0.8)
            torch.cuda.set_per_process_memory_fraction(memory_fraction)
            
            # Enable cuDNN benchmarking for faster convolutions
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            
            logger.info(f"GPU optimization enabled: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            # CPU optimizations
            num_threads = self.config.get('num_threads', 4)
            torch.set_num_threads(num_threads)
            logger.info(f"CPU optimization enabled: {num_threads} threads")
        
        # Enable TensorFloat32 for faster computation on Ampere GPUs
        if self.use_gpu and torch.cuda.get_device_capability()[0] >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("TensorFloat32 enabled")
    
    def optimize_model(self, model):
        """Apply optimizations to a model"""
        if self.use_gpu:
            model = model.cuda()
        
        # Set to evaluation mode
        model.eval()
        
        # Enable inference mode optimizations
        torch.set_grad_enabled(False)
        
        return model
    
    def get_device(self) -> str:
        """Get the device to use"""
        return 'cuda' if self.use_gpu else 'cpu'
