"""
Cursor control using hand gestures - Enhanced with Kalman filter smoothing
Supports: cursor movement, click, double-click, right-click, drag, scroll
"""

import pyautogui
import numpy as np
import logging
from typing import Optional, Tuple, List
import time

logger = logging.getLogger(__name__)


class KalmanFilter1D:
    """Simple 1D Kalman filter for cursor smoothing"""
    
    def __init__(self, process_variance=0.01, measurement_variance=0.1):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        
        # Initial state
        self.estimate = 0
        self.error_estimate = 1
    
    def update(self, measurement: float) -> float:
        """Update filter with new measurement and return estimate"""
        # Prediction
        prediction = self.estimate
        error_prediction = self.error_estimate + self.process_variance
        
        # Kalman gain
        kalman_gain = error_prediction / (error_prediction + self.measurement_variance)
        
        # Update estimate
        self.estimate = prediction + kalman_gain * (measurement - prediction)
        self.error_estimate = (1 - kalman_gain) * error_prediction
        
        return self.estimate
    
    def reset(self, value: float):
        """Reset filter to a specific value"""
        self.estimate = value
        self.error_estimate = 1


class CursorController:
    """Control system cursor with hand gestures - Enhanced version"""
    
    def __init__(self, screen_width: int = 1920, screen_height: int = 1080,
                 smoothing_algorithm: str = 'kalman'):
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Smoothing algorithm selection
        self.smoothing_algorithm = smoothing_algorithm
        
        # Kalman filters for X and Y
        self.kalman_x = KalmanFilter1D(process_variance=0.005, measurement_variance=0.1)
        self.kalman_y = KalmanFilter1D(process_variance=0.005, measurement_variance=0.1)
        self.kalman_initialized = False
        
        # Average smoothing parameters (fallback)
        self.smoothing = 5
        self.prev_positions: List[Tuple[int, int]] = []
        
        # Exponential smoothing
        self.exp_alpha = 0.3  # Lower = smoother
        self.exp_x = None
        self.exp_y = None
        
        # Click state
        self.click_cooldown = 0.25  # seconds
        self.last_click_time = 0
        self.last_right_click_time = 0
        
        # Movement bounds
        self.dead_zone = 5  # pixels - reduced for responsiveness
        
        # Sensitivity (multiplier for cursor movement range)
        self.sensitivity = 1.2
        
        # Drag state
        self.dragging = False
        self.drag_button = 'left'
        
        # Get actual screen size
        try:
            self.screen_width, self.screen_height = pyautogui.size()
            logger.info(f"Screen size: {self.screen_width}x{self.screen_height}")
        except Exception as e:
            logger.warning(f"Could not get screen size: {e}")
        
        # Disable pyautogui failsafe for smooth operation
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0.005  # Minimal pause for speed
    
    def map_to_screen(self, hand_x: int, hand_y: int, 
                      frame_width: int, frame_height: int) -> Tuple[int, int]:
        """Map hand position from camera frame to screen coordinates"""
        # Flip x-axis for mirror effect
        hand_x = frame_width - hand_x
        
        # Apply sensitivity
        # Expand from center of frame
        center_x = frame_width / 2
        center_y = frame_height / 2
        
        # Scale from center with sensitivity
        scaled_x = center_x + (hand_x - center_x) * self.sensitivity
        scaled_y = center_y + (hand_y - center_y) * self.sensitivity
        
        # Map to screen coordinates
        screen_x = int((scaled_x / frame_width) * self.screen_width)
        screen_y = int((scaled_y / frame_height) * self.screen_height)
        
        # Clamp to screen bounds
        screen_x = max(0, min(screen_x, self.screen_width - 1))
        screen_y = max(0, min(screen_y, self.screen_height - 1))
        
        return screen_x, screen_y
    
    def smooth_position(self, x: int, y: int) -> Tuple[int, int]:
        """Apply selected smoothing algorithm to cursor movement"""
        if self.smoothing_algorithm == 'kalman':
            return self._smooth_kalman(x, y)
        elif self.smoothing_algorithm == 'exponential':
            return self._smooth_exponential(x, y)
        else:
            return self._smooth_average(x, y)
    
    def _smooth_kalman(self, x: int, y: int) -> Tuple[int, int]:
        """Apply Kalman filter smoothing"""
        if not self.kalman_initialized:
            self.kalman_x.reset(x)
            self.kalman_y.reset(y)
            self.kalman_initialized = True
            return x, y
        
        smooth_x = int(self.kalman_x.update(x))
        smooth_y = int(self.kalman_y.update(y))
        
        return smooth_x, smooth_y
    
    def _smooth_exponential(self, x: int, y: int) -> Tuple[int, int]:
        """Apply exponential smoothing"""
        if self.exp_x is None:
            self.exp_x = x
            self.exp_y = y
            return x, y
        
        self.exp_x = self.exp_alpha * x + (1 - self.exp_alpha) * self.exp_x
        self.exp_y = self.exp_alpha * y + (1 - self.exp_alpha) * self.exp_y
        
        return int(self.exp_x), int(self.exp_y)
    
    def _smooth_average(self, x: int, y: int) -> Tuple[int, int]:
        """Apply moving average smoothing"""
        self.prev_positions.append((x, y))
        
        # Keep only last N positions
        if len(self.prev_positions) > self.smoothing:
            self.prev_positions.pop(0)
        
        # Calculate average
        avg_x = int(np.mean([pos[0] for pos in self.prev_positions]))
        avg_y = int(np.mean([pos[1] for pos in self.prev_positions]))
        
        return avg_x, avg_y
    
    def move_cursor(self, hand_x: int, hand_y: int, 
                    frame_width: int, frame_height: int):
        """Move system cursor to hand position"""
        try:
            # Map to screen coordinates
            screen_x, screen_y = self.map_to_screen(hand_x, hand_y, frame_width, frame_height)
            
            # Apply smoothing
            smooth_x, smooth_y = self.smooth_position(screen_x, screen_y)
            
            # Get current cursor position
            current_x, current_y = pyautogui.position()
            
            # Only move if beyond dead zone
            distance = np.sqrt((smooth_x - current_x)**2 + (smooth_y - current_y)**2)
            
            if distance > self.dead_zone:
                pyautogui.moveTo(smooth_x, smooth_y, duration=0)
                
        except Exception as e:
            logger.error(f"Cursor movement failed: {e}")
    
    def click(self, button: str = 'left'):
        """Perform mouse click with cooldown"""
        try:
            current_time = time.time()
            
            # Check cooldown
            if current_time - self.last_click_time < self.click_cooldown:
                return False
            
            pyautogui.click(button=button)
            self.last_click_time = current_time
            logger.debug(f"{button} click performed")
            return True
            
        except Exception as e:
            logger.error(f"Click failed: {e}")
            return False
    
    def double_click(self, button: str = 'left'):
        """Perform double click"""
        try:
            pyautogui.doubleClick(button=button)
            self.last_click_time = time.time()
            logger.debug(f"{button} double-click performed")
            return True
        except Exception as e:
            logger.error(f"Double-click failed: {e}")
            return False
    
    def right_click(self):
        """Perform right click with cooldown"""
        try:
            current_time = time.time()
            
            # Longer cooldown for right-click to prevent accidental triggers
            if current_time - self.last_right_click_time < self.click_cooldown * 2:
                return False
            
            pyautogui.click(button='right')
            self.last_right_click_time = current_time
            logger.debug("Right click performed")
            return True
            
        except Exception as e:
            logger.error(f"Right-click failed: {e}")
            return False
    
    def scroll(self, amount: int):
        """Scroll mouse wheel"""
        try:
            if amount != 0:
                pyautogui.scroll(amount)
                logger.debug(f"Scrolled {amount}")
        except Exception as e:
            logger.error(f"Scroll failed: {e}")
    
    def start_drag(self, button: str = 'left'):
        """Start a drag operation"""
        try:
            if not self.dragging:
                pyautogui.mouseDown(button=button)
                self.dragging = True
                self.drag_button = button
                logger.debug(f"Started drag with {button} button")
        except Exception as e:
            logger.error(f"Start drag failed: {e}")
    
    def end_drag(self):
        """End a drag operation"""
        try:
            if self.dragging:
                pyautogui.mouseUp(button=self.drag_button)
                self.dragging = False
                logger.debug("Ended drag")
        except Exception as e:
            logger.error(f"End drag failed: {e}")
    
    def drag_to(self, x: int, y: int):
        """Move cursor while dragging"""
        if self.dragging:
            try:
                pyautogui.moveTo(x, y, duration=0)
            except Exception as e:
                logger.error(f"Drag move failed: {e}")
    
    def handle_gesture_input(self, gesture_data: dict, frame_width: int, frame_height: int):
        """
        Handle all gesture inputs from hand detector
        
        Args:
            gesture_data: Dictionary from HandGestureDetector.detect()
            frame_width: Camera frame width
            frame_height: Camera frame height
        """
        cursor_pos = gesture_data.get('cursor_position')
        click_detected = gesture_data.get('click_detected', False)
        double_click_detected = gesture_data.get('double_click_detected', False)
        right_click_detected = gesture_data.get('right_click_detected', False)
        scroll_amount = gesture_data.get('scroll_amount', 0)
        drag_active = gesture_data.get('drag_active', False)
        
        # Move cursor
        if cursor_pos:
            self.move_cursor(cursor_pos[0], cursor_pos[1], frame_width, frame_height)
        
        # Handle drag state
        if drag_active and not self.dragging:
            self.start_drag()
        elif not drag_active and self.dragging:
            self.end_drag()
        
        # Handle clicks (only when not dragging)
        if not self.dragging:
            if double_click_detected:
                self.double_click()
            elif right_click_detected:
                self.right_click()
            elif click_detected:
                self.click()
        
        # Handle scroll
        if scroll_amount != 0:
            self.scroll(scroll_amount)
    
    def reset(self):
        """Reset all state"""
        self.kalman_initialized = False
        self.prev_positions = []
        self.exp_x = None
        self.exp_y = None
        self.dragging = False
        if self.dragging:
            self.end_drag()
    
    def set_sensitivity(self, sensitivity: float):
        """Set cursor movement sensitivity (default 1.2)"""
        self.sensitivity = max(0.5, min(3.0, sensitivity))
        logger.info(f"Cursor sensitivity set to {self.sensitivity}")
    
    def set_smoothing_algorithm(self, algorithm: str):
        """Set smoothing algorithm ('kalman', 'exponential', 'average')"""
        if algorithm in ['kalman', 'exponential', 'average']:
            self.smoothing_algorithm = algorithm
            self.reset()
            logger.info(f"Smoothing algorithm set to {algorithm}")
