"""
Hand gesture detection and recognition - Enhanced with more gestures
Supports: cursor control, click, double-click, right-click, drag, scroll
"""

import cv2
import numpy as np
import logging
import time
from typing import Dict, List, Optional, Tuple
from collections import deque

logger = logging.getLogger(__name__)


class HandGestureDetector:
    """Detect and recognize hand gestures for control"""
    
    # Gesture mapping for commands
    GESTURE_COMMANDS = {
        'fist': 'grab',
        'open': 'release',
        'pointing': 'cursor',
        'peace': 'scroll',
        'pinch': 'click',
        'double_pinch': 'double_click',
        'three_finger_pinch': 'right_click',
        'thumbs_up': 'confirm',
        'thumbs_down': 'cancel',
        'rock': 'special'
    }
    
    def __init__(self, config: Dict, optimizer):
        self.config = config
        self.optimizer = optimizer
        self.detector = None
        self.active_detector = None
        
        # Gesture configuration
        self.click_threshold = config.get('click_threshold', 30)  # pixels
        self.double_click_window = config.get('double_click_window', 0.5)  # seconds
        self.pinch_threshold = config.get('pinch_threshold', 0.05)  # normalized distance
        
        # Gesture history for smoothing
        self.gesture_history = deque(maxlen=5)
        self.position_history = deque(maxlen=10)
        
        # Click state tracking
        self.last_click_time = 0
        self.click_count = 0
        self.pinch_state = False
        self.prev_pinch_state = False
        
        # Scroll state
        self.scroll_start_y = None
        self.scroll_active = False
        
        # Drag state
        self.drag_active = False
        self.drag_start_pos = None
        
        self._load_detector()
    
    def _load_detector(self):
        """Load hand detection model"""
        try:
            import mediapipe as mp
            self.mp_hands = mp.solutions.hands
            self.mp_drawing = mp.solutions.drawing_utils
            
            self.detector = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.5
            )
            self.active_detector = 'mediapipe'
            logger.info("MediaPipe hand detection loaded")
            
        except Exception as e:
            logger.warning(f"MediaPipe failed: {e}")
            self._load_opencv_fallback()
    
    def _load_opencv_fallback(self):
        """Load OpenCV-based hand detection as fallback"""
        try:
            self.detector = 'opencv_skin'
            self.active_detector = 'opencv'
            logger.info("OpenCV skin detection loaded as fallback")
        except Exception as e:
            logger.error(f"All hand detectors failed: {e}")
    
    def detect(self, frame: np.ndarray) -> Dict:
        """
        Detect hands and gestures
        Returns:
        {
            'hands': [{
                'landmarks': [...],
                'handedness': 'Left'/'Right',
                'bbox': [x1, y1, x2, y2],
                'gesture': 'pointing'/'fist'/'open'/'pinch'/etc.
            }],
            'cursor_position': (x, y),
            'click_detected': bool,
            'double_click_detected': bool,
            'right_click_detected': bool,
            'scroll_amount': int (positive=up, negative=down),
            'drag_active': bool,
            'drag_start': (x, y) or None,
            'gesture_command': str
        }
        """
        if self.detector is None:
            return self._empty_result()
        
        try:
            if self.active_detector == 'mediapipe':
                return self._detect_mediapipe(frame)
            elif self.active_detector == 'opencv':
                return self._detect_opencv(frame)
        except Exception as e:
            logger.error(f"Hand detection failed: {e}")
            return self._empty_result()
    
    def _empty_result(self):
        """Return empty result structure"""
        return {
            'hands': [],
            'cursor_position': None,
            'click_detected': False,
            'double_click_detected': False,
            'right_click_detected': False,
            'scroll_amount': 0,
            'drag_active': False,
            'drag_start': None,
            'gesture_command': None
        }
    
    def _detect_mediapipe(self, frame: np.ndarray) -> Dict:
        """Detect hands using MediaPipe"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb_frame)
        
        h, w = frame.shape[:2]
        hands_data = []
        cursor_pos = None
        click_detected = False
        double_click_detected = False
        right_click_detected = False
        scroll_amount = 0
        gesture_command = None
        
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Get handedness
                handedness = results.multi_handedness[idx].classification[0].label
                
                # Extract landmarks
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.append({
                        'x': lm.x * w,
                        'y': lm.y * h,
                        'z': lm.z
                    })
                
                # Calculate bounding box
                x_coords = [lm['x'] for lm in landmarks]
                y_coords = [lm['y'] for lm in landmarks]
                bbox = [
                    int(min(x_coords)),
                    int(min(y_coords)),
                    int(max(x_coords)),
                    int(max(y_coords))
                ]
                
                # Recognize gesture
                gesture = self._recognize_gesture(landmarks)
                
                # Smooth gesture recognition
                self.gesture_history.append(gesture)
                stable_gesture = self._get_stable_gesture()
                
                # Get cursor position from index finger tip (landmark 8)
                index_tip = landmarks[8]
                cursor_pos = (int(index_tip['x']), int(index_tip['y']))
                
                # Add to position history for smoothing
                self.position_history.append(cursor_pos)
                
                # Detect pinch (thumb and index finger)
                thumb_tip = landmarks[4]
                pinch_distance = self._calculate_distance(thumb_tip, index_tip, w, h)
                current_pinch = pinch_distance < self.pinch_threshold
                
                # Detect click on pinch start (not during hold)
                if current_pinch and not self.prev_pinch_state:
                    current_time = time.time()
                    
                    if current_time - self.last_click_time < self.double_click_window:
                        double_click_detected = True
                        self.click_count = 0
                    else:
                        click_detected = True
                        self.click_count = 1
                    
                    self.last_click_time = current_time
                
                self.prev_pinch_state = current_pinch
                self.pinch_state = current_pinch
                
                # Detect three-finger pinch for right-click
                middle_tip = landmarks[12]
                three_finger_distance = (
                    self._calculate_distance(thumb_tip, index_tip, w, h) +
                    self._calculate_distance(thumb_tip, middle_tip, w, h)
                ) / 2
                
                if three_finger_distance < self.pinch_threshold * 1.5:
                    if stable_gesture == 'three_finger_pinch':
                        right_click_detected = True
                
                # Detect scroll gesture (peace sign moving up/down)
                if stable_gesture == 'peace':
                    if self.scroll_start_y is None:
                        self.scroll_start_y = cursor_pos[1]
                        self.scroll_active = True
                    else:
                        y_delta = self.scroll_start_y - cursor_pos[1]
                        if abs(y_delta) > 20:  # Minimum movement threshold
                            scroll_amount = int(y_delta / 20)  # Scale to scroll units
                            self.scroll_start_y = cursor_pos[1]
                else:
                    self.scroll_start_y = None
                    self.scroll_active = False
                
                # Detect drag gesture (fist)
                if stable_gesture == 'fist':
                    if not self.drag_active:
                        self.drag_active = True
                        self.drag_start_pos = cursor_pos
                else:
                    self.drag_active = False
                    self.drag_start_pos = None
                
                # Determine gesture command
                gesture_command = self._get_gesture_command(stable_gesture, landmarks)
                
                hands_data.append({
                    'landmarks': landmarks,
                    'handedness': handedness,
                    'bbox': bbox,
                    'gesture': stable_gesture,
                    'pinch_distance': pinch_distance
                })
        else:
            # No hands detected - reset states
            self.scroll_start_y = None
            self.drag_active = False
            self.drag_start_pos = None
        
        return {
            'hands': hands_data,
            'cursor_position': cursor_pos,
            'click_detected': click_detected,
            'double_click_detected': double_click_detected,
            'right_click_detected': right_click_detected,
            'scroll_amount': scroll_amount,
            'drag_active': self.drag_active,
            'drag_start': self.drag_start_pos,
            'gesture_command': gesture_command
        }
    
    def _calculate_distance(self, point1: Dict, point2: Dict, 
                           frame_width: int, frame_height: int) -> float:
        """Calculate normalized distance between two landmarks"""
        # Normalize by frame diagonal
        diagonal = np.sqrt(frame_width**2 + frame_height**2)
        
        dx = (point1['x'] - point2['x'])
        dy = (point1['y'] - point2['y'])
        distance = np.sqrt(dx**2 + dy**2)
        
        return distance / diagonal
    
    def _recognize_gesture(self, landmarks: List[Dict]) -> str:
        """Recognize hand gesture from landmarks"""
        # Get finger tip and base positions
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        thumb_mcp = landmarks[2]
        
        index_tip = landmarks[8]
        index_pip = landmarks[6]
        index_mcp = landmarks[5]
        
        middle_tip = landmarks[12]
        middle_pip = landmarks[10]
        
        ring_tip = landmarks[16]
        ring_pip = landmarks[14]
        
        pinky_tip = landmarks[20]
        pinky_pip = landmarks[18]
        
        wrist = landmarks[0]
        
        # Calculate if fingers are extended (tip above pip joint)
        def is_finger_extended(tip, pip):
            return tip['y'] < pip['y']
        
        index_extended = is_finger_extended(index_tip, index_pip)
        middle_extended = is_finger_extended(middle_tip, middle_pip)
        ring_extended = is_finger_extended(ring_tip, ring_pip)
        pinky_extended = is_finger_extended(pinky_tip, pinky_pip)
        
        # Thumb detection (horizontal movement)
        thumb_extended = thumb_tip['x'] < thumb_ip['x']  # For right hand
        
        # Calculate pinch distance
        pinch_dist = np.sqrt(
            (thumb_tip['x'] - index_tip['x'])**2 + 
            (thumb_tip['y'] - index_tip['y'])**2
        )
        
        # Three-finger pinch detection
        three_finger_dist = np.sqrt(
            (thumb_tip['x'] - middle_tip['x'])**2 + 
            (thumb_tip['y'] - middle_tip['y'])**2
        )
        
        # Count extended fingers
        extended_count = sum([
            index_extended,
            middle_extended,
            ring_extended,
            pinky_extended
        ])
        
        # Recognize specific gestures
        
        # Pinch: thumb and index close together
        if pinch_dist < 40 and not middle_extended:
            return 'pinch'
        
        # Three-finger pinch: thumb, index, and middle close
        if pinch_dist < 50 and three_finger_dist < 50:
            return 'three_finger_pinch'
        
        # Fist: no fingers extended
        if extended_count == 0 and not thumb_extended:
            return 'fist'
        
        # Open palm: all fingers extended
        if extended_count == 4 and thumb_extended:
            return 'open'
        
        # Pointing: only index extended
        if index_extended and not middle_extended and not ring_extended and not pinky_extended:
            return 'pointing'
        
        # Peace: index and middle extended
        if index_extended and middle_extended and not ring_extended and not pinky_extended:
            return 'peace'
        
        # Rock: index and pinky extended
        if index_extended and pinky_extended and not middle_extended and not ring_extended:
            return 'rock'
        
        # Thumbs up: only thumb extended, hand vertical
        if thumb_extended and extended_count == 0:
            if thumb_tip['y'] < wrist['y']:
                return 'thumbs_up'
            else:
                return 'thumbs_down'
        
        return 'unknown'
    
    def _get_stable_gesture(self) -> str:
        """Get most common gesture from history for stability"""
        if not self.gesture_history:
            return 'unknown'
        
        # Count occurrences
        gesture_counts = {}
        for gesture in self.gesture_history:
            gesture_counts[gesture] = gesture_counts.get(gesture, 0) + 1
        
        # Return most common
        return max(gesture_counts, key=gesture_counts.get)
    
    def _get_gesture_command(self, gesture: str, landmarks: List[Dict]) -> Optional[str]:
        """Map gesture to command"""
        return self.GESTURE_COMMANDS.get(gesture)
    
    def _detect_opencv(self, frame: np.ndarray) -> Dict:
        """Detect hands using OpenCV skin detection (fallback)"""
        # Convert to HSV for skin detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define skin color range
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Create mask
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        hands_data = []
        cursor_pos = None
        
        if contours:
            # Get largest contour (assume it's the hand)
            largest_contour = max(contours, key=cv2.contourArea)
            
            if cv2.contourArea(largest_contour) > 5000:
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Cursor position at top center of hand
                cursor_pos = (x + w // 2, y)
                
                # Simple gesture detection based on contour
                hull = cv2.convexHull(largest_contour)
                hull_area = cv2.contourArea(hull)
                contour_area = cv2.contourArea(largest_contour)
                
                solidity = contour_area / hull_area if hull_area > 0 else 0
                
                # High solidity = fist, low solidity = open hand
                if solidity > 0.9:
                    gesture = 'fist'
                elif solidity > 0.7:
                    gesture = 'pointing'
                else:
                    gesture = 'open'
                
                hands_data.append({
                    'landmarks': [],
                    'handedness': 'Unknown',
                    'bbox': [x, y, x + w, y + h],
                    'gesture': gesture
                })
        
        return {
            'hands': hands_data,
            'cursor_position': cursor_pos,
            'click_detected': False,
            'double_click_detected': False,
            'right_click_detected': False,
            'scroll_amount': 0,
            'drag_active': False,
            'drag_start': None,
            'gesture_command': 'cursor' if cursor_pos else None
        }
    
    def get_smoothed_cursor(self) -> Optional[Tuple[int, int]]:
        """Get smoothed cursor position from history"""
        if not self.position_history:
            return None
        
        positions = list(self.position_history)
        avg_x = int(np.mean([p[0] for p in positions]))
        avg_y = int(np.mean([p[1] for p in positions]))
        
        return (avg_x, avg_y)
