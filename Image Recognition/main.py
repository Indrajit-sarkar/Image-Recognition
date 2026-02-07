"""
Advanced Multi-Modal Image Recognition System
High-speed, high-accuracy detection with multiple backup mechanisms
"""

import cv2
import numpy as np
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

from detectors.object_detector import ObjectDetector
from detectors.depth_estimator import DepthEstimator
from detectors.ocr_engine import OCREngine
from detectors.face_eye_tracker import FaceEyeTracker
from detectors.currency_detector import CurrencyDetector
from detectors.medicine_detector import MedicineDetector
from detectors.food_detector import FoodDetector
from detectors.hand_gesture_detector import HandGestureDetector
from utils.camera_manager import CameraManager
from utils.performance_optimizer import PerformanceOptimizer
from utils.result_aggregator import ResultAggregator
from utils.visualization import Visualizer
from utils.cursor_controller import CursorController
from utils.structure_3d_creator import Structure3DCreator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedImageRecognition:
    """Main class for advanced image recognition system"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the recognition system"""
        logger.info("Initializing Advanced Image Recognition System...")
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize performance optimizer
        self.optimizer = PerformanceOptimizer(self.config)
        
        # Initialize camera
        self.camera = CameraManager(self.config['camera'])
        
        # Initialize all detectors with backup mechanisms
        self._initialize_detectors()
        
        # Initialize utilities
        self.aggregator = ResultAggregator()
        self.visualizer = Visualizer(self.config)
        self.cursor_controller = CursorController()
        self.structure_3d = Structure3DCreator()
        
        # Gesture control mode
        self.gesture_mode = 'detection'  # 'detection', 'cursor', '3d'
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(
            max_workers=self.config['performance']['num_threads']
        )
        
        logger.info("System initialized successfully!")
    
    def _initialize_detectors(self):
        """Initialize all detection modules with fallback support"""
        try:
            self.object_detector = ObjectDetector(
                self.config['models']['object_detection'],
                self.optimizer
            )
            logger.info("✓ Object detector initialized")
        except Exception as e:
            logger.error(f"Object detector failed: {e}")
            raise
        
        try:
            self.depth_estimator = DepthEstimator(
                self.config['models']['depth_estimation'],
                self.optimizer
            )
            logger.info("✓ Depth estimator initialized")
        except Exception as e:
            logger.warning(f"Depth estimator failed: {e}")
            self.depth_estimator = None
        
        try:
            self.ocr_engine = OCREngine(
                self.config['models']['ocr'],
                self.optimizer
            )
            logger.info("✓ OCR engine initialized")
        except Exception as e:
            logger.warning(f"OCR engine failed: {e}")
            self.ocr_engine = None
        
        try:
            self.face_eye_tracker = FaceEyeTracker(
                self.config['models']['face_detection'],
                self.config['models']['eye_tracking'],
                self.optimizer
            )
            logger.info("✓ Face/Eye tracker initialized")
        except Exception as e:
            logger.warning(f"Face/Eye tracker failed: {e}")
            self.face_eye_tracker = None
        
        try:
            self.currency_detector = CurrencyDetector(
                self.config['models']['currency_detection'],
                self.optimizer
            )
            logger.info("✓ Currency detector initialized")
        except Exception as e:
            logger.warning(f"Currency detector failed: {e}")
            self.currency_detector = None
        
        try:
            self.medicine_detector = MedicineDetector(
                self.config['models']['medicine_detection'],
                self.optimizer
            )
            logger.info("✓ Medicine detector initialized")
        except Exception as e:
            logger.warning(f"Medicine detector failed: {e}")
            self.medicine_detector = None
        
        try:
            self.food_detector = FoodDetector(
                self.config['models']['food_detection'],
                self.optimizer
            )
            logger.info("✓ Food detector initialized")
        except Exception as e:
            logger.warning(f"Food detector failed: {e}")
            self.food_detector = None
        
        try:
            self.hand_gesture_detector = HandGestureDetector(
                self.config['models'].get('hand_detection', {}),
                self.optimizer
            )
            logger.info("✓ Hand gesture detector initialized")
        except Exception as e:
            logger.warning(f"Hand gesture detector failed: {e}")
            self.hand_gesture_detector = None
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """Process a single frame with all detectors in parallel"""
        start_time = time.time()
        
        # Submit all detection tasks in parallel
        futures = {}
        
        # Core object detection (always runs)
        futures['objects'] = self.executor.submit(
            self.object_detector.detect, frame
        )
        
        # Depth estimation
        if self.depth_estimator:
            futures['depth'] = self.executor.submit(
                self.depth_estimator.estimate, frame
            )
        
        # OCR for text detection
        if self.ocr_engine:
            futures['text'] = self.executor.submit(
                self.ocr_engine.detect_text, frame
            )
        
        # Face and eye tracking
        if self.face_eye_tracker:
            futures['faces'] = self.executor.submit(
                self.face_eye_tracker.detect, frame
            )
        
        # Currency detection
        if self.currency_detector:
            futures['currency'] = self.executor.submit(
                self.currency_detector.detect, frame
            )
        
        # Medicine detection
        if self.medicine_detector:
            futures['medicine'] = self.executor.submit(
                self.medicine_detector.detect, frame
            )
        
        # Food detection
        if self.food_detector:
            futures['food'] = self.executor.submit(
                self.food_detector.detect, frame
            )
        
        # Hand gesture detection
        if self.hand_gesture_detector:
            futures['hands'] = self.executor.submit(
                self.hand_gesture_detector.detect, frame
            )
        
        # Collect results
        results = {}
        for key, future in futures.items():
            try:
                results[key] = future.result(timeout=1.0)
            except Exception as e:
                # Silently handle failures, just return None
                results[key] = None
        
        # Aggregate and post-process results
        aggregated = self.aggregator.aggregate(results)
        
        # Add timing information
        aggregated['processing_time'] = time.time() - start_time
        aggregated['fps'] = 1.0 / aggregated['processing_time']
        
        return aggregated
    
    def run_realtime(self, camera_source: Optional[int] = None):
        """Run real-time detection from camera"""
        logger.info("Starting real-time detection...")
        
        if camera_source is not None:
            self.camera.set_source(camera_source)
        
        try:
            while True:
                # Capture frame
                frame = self.camera.read_frame()
                if frame is None:
                    logger.warning("Failed to read frame")
                    continue
                
                # Process frame
                results = self.process_frame(frame)
                
                # Add gesture mode to results for visualization
                results['gesture_mode'] = self.gesture_mode
                
                # Handle hand gestures
                if results.get('hands') and self.hand_gesture_detector:
                    self._handle_hand_gestures(results['hands'], frame.shape)
                
                # Visualize results
                output_frame = self.visualizer.draw_results(frame, results)
                
                # Render 3D structures if in 3D mode
                if self.gesture_mode == '3d':
                    output_frame = self.structure_3d.render(output_frame)
                
                # Display
                cv2.imshow('Advanced Image Recognition', output_frame)
                
                # Display FPS and stats on the frame itself (displayOverlay not supported)
                # Already shown in visualization
                
                # Break on 'q' key
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('m'):
                    # Toggle mode
                    self._toggle_mode()
                elif key == ord('c'):
                    # Create cube
                    self.structure_3d.create_shape('cube')
                elif key == ord('p'):
                    # Create pyramid
                    self.structure_3d.create_shape('pyramid')
                elif key == ord('s'):
                    # Create sphere
                    self.structure_3d.create_shape('sphere')
                elif key == ord('y'):
                    # Create cylinder
                    self.structure_3d.create_shape('cylinder')
                elif key == ord('x'):
                    # Clear all 3D structures
                    self.structure_3d.clear_all()
                
        except KeyboardInterrupt:
            logger.info("Stopping detection...")
        finally:
            self.camera.release()
            cv2.destroyAllWindows()
    
    def process_image(self, image_path: str) -> Dict:
        """Process a single image file"""
        logger.info(f"Processing image: {image_path}")
        
        # Load image
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Process
        results = self.process_frame(frame)
        
        # Visualize and save
        output_frame = self.visualizer.draw_results(frame, results)
        output_path = Path(self.config['output']['results_dir']) / f"result_{Path(image_path).name}"
        output_path.parent.mkdir(exist_ok=True)
        cv2.imwrite(str(output_path), output_frame)
        
        logger.info(f"Results saved to: {output_path}")
        return results
    
    def process_video(self, video_path: str):
        """Process a video file"""
        logger.info(f"Processing video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup output video
        output_path = Path(self.config['output']['results_dir']) / f"result_{Path(video_path).name}"
        output_path.parent.mkdir(exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                results = self.process_frame(frame)
                
                # Visualize
                output_frame = self.visualizer.draw_results(frame, results)
                
                # Write to output
                out.write(output_frame)
                
                # Display progress
                cv2.imshow('Processing Video', output_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            cap.release()
            out.release()
            cv2.destroyAllWindows()
        
        logger.info(f"Video processed and saved to: {output_path}")
    
    def cleanup(self):
        """Cleanup resources"""
        self.executor.shutdown(wait=True)
        self.camera.release()
        logger.info("System cleanup completed")
    
    def _toggle_mode(self):
        """Toggle between detection modes"""
        modes = ['detection', 'cursor', '3d']
        current_idx = modes.index(self.gesture_mode)
        self.gesture_mode = modes[(current_idx + 1) % len(modes)]
        logger.info(f"Switched to {self.gesture_mode} mode")
    
    def _handle_hand_gestures(self, hand_data: dict, frame_shape: tuple):
        """Handle hand gesture commands - Enhanced with new gestures"""
        if not hand_data:
            return
        
        h, w = frame_shape[:2]
        cursor_pos = hand_data.get('cursor_position')
        click_detected = hand_data.get('click_detected', False)
        double_click_detected = hand_data.get('double_click_detected', False)
        right_click_detected = hand_data.get('right_click_detected', False)
        scroll_amount = hand_data.get('scroll_amount', 0)
        drag_active = hand_data.get('drag_active', False)
        gesture_command = hand_data.get('gesture_command')
        
        # Cursor control mode - Use enhanced handler
        if self.gesture_mode == 'cursor':
            self.cursor_controller.handle_gesture_input(hand_data, w, h)
        
        # 3D structure mode
        elif self.gesture_mode == '3d':
            if gesture_command and cursor_pos:
                self.structure_3d.handle_gesture(gesture_command, cursor_pos)
                
                # Rotate structure with hand movement
                if gesture_command == 'grab' and self.structure_3d.current_structure:
                    self.structure_3d.rotate('y', 2)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Advanced Image Recognition System')
    parser.add_argument('--mode', choices=['realtime', 'image', 'video'], 
                       default='realtime', help='Processing mode')
    parser.add_argument('--input', type=str, help='Input file path (for image/video mode)')
    parser.add_argument('--camera', type=int, default=0, help='Camera source (0=built-in, 1+=external)')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    
    args = parser.parse_args()
    
    # Initialize system
    system = AdvancedImageRecognition(args.config)
    
    try:
        if args.mode == 'realtime':
            system.run_realtime(args.camera)
        elif args.mode == 'image':
            if not args.input:
                raise ValueError("--input required for image mode")
            results = system.process_image(args.input)
            print(f"\nDetection Results:\n{results}")
        elif args.mode == 'video':
            if not args.input:
                raise ValueError("--input required for video mode")
            system.process_video(args.input)
    finally:
        system.cleanup()


if __name__ == "__main__":
    main()
