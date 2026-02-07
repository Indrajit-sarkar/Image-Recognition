"""
Visualization utilities for drawing detection results
"""

import cv2
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class Visualizer:
    """Visualize detection results on frames"""
    
    def __init__(self, config: Dict):
        self.config = config['output']
        
        # Colors for different detection types
        self.colors = {
            'object': (0, 255, 0),      # Green
            'face': (255, 0, 0),        # Blue
            'eye': (0, 255, 255),       # Yellow
            'text': (255, 255, 0),      # Cyan
            'currency': (0, 165, 255),  # Orange
            'medicine': (255, 0, 255),  # Magenta
            'food': (0, 128, 255)       # Orange-red
        }
    
    def draw_results(self, frame: np.ndarray, results: Dict) -> np.ndarray:
        """Draw all detection results on frame"""
        output = frame.copy()
        
        # Draw depth map overlay if available
        if results.get('depth_map') is not None and self.config.get('show_distance', True):
            output = self._draw_depth_overlay(output, results['depth_map'])
        
        # Draw objects
        if results.get('objects'):
            output = self._draw_objects(output, results['objects'])
        
        # Draw text regions
        if results.get('text'):
            output = self._draw_text_regions(output, results['text'])
        
        # Draw faces
        if results.get('faces'):
            output = self._draw_faces(output, results['faces'])
        
        # Draw eyes
        if results.get('eyes'):
            output = self._draw_eyes(output, results['eyes'])
        
        # Draw currency
        if results.get('currency'):
            output = self._draw_currency(output, results['currency'])
        
        # Draw medicine
        if results.get('medicine'):
            output = self._draw_medicine(output, results['medicine'])
        
        # Draw food
        if results.get('food'):
            output = self._draw_food(output, results['food'])
        
        # Draw hand gestures
        if results.get('hands'):
            output = self._draw_hands(output, results['hands'])
        
        # Draw summary info
        if results.get('summary'):
            output = self._draw_summary(output, results['summary'], results.get('fps', 0), 
                                       results.get('gesture_mode', 'detection'), results.get('hands'))
        
        return output
    
    def _draw_depth_overlay(self, frame: np.ndarray, depth_map: np.ndarray) -> np.ndarray:
        """Draw depth map as semi-transparent overlay"""
        # Convert depth to color map
        depth_colored = cv2.applyColorMap(
            (depth_map * 255).astype(np.uint8), 
            cv2.COLORMAP_PLASMA
        )
        
        # Blend with original frame
        alpha = 0.3
        output = cv2.addWeighted(frame, 1 - alpha, depth_colored, alpha, 0)
        
        return output
    
    def _draw_objects(self, frame: np.ndarray, objects: List[Dict]) -> np.ndarray:
        """Draw object detections"""
        for obj in objects:
            bbox = obj.get('bbox')
            if not bbox:
                continue
            
            x1, y1, x2, y2 = bbox
            class_name = obj.get('class', 'unknown')
            confidence = obj.get('confidence', 0)
            distance = obj.get('distance')
            
            # Draw bounding box
            if self.config.get('show_bounding_boxes', True):
                cv2.rectangle(frame, (x1, y1), (x2, y2), self.colors['object'], 2)
            
            # Draw label
            if self.config.get('show_labels', True):
                label = f"{class_name}"
                
                if self.config.get('show_confidence', True):
                    label += f" {confidence:.2f}"
                
                if distance is not None and self.config.get('show_distance', True):
                    label += f" {distance:.1f}m"
                
                # Background for text
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), self.colors['object'], -1)
                cv2.putText(frame, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return frame
    
    def _draw_text_regions(self, frame: np.ndarray, text_regions: List[Dict]) -> np.ndarray:
        """Draw text detection regions"""
        for region in text_regions:
            bbox = region.get('bbox')
            if not bbox:
                continue
            
            x1, y1, x2, y2 = bbox
            text = region.get('text', '')
            confidence = region.get('confidence', 0)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.colors['text'], 2)
            
            # Draw text
            label = f"Text: {text[:20]}"
            if self.config.get('show_confidence', True):
                label += f" ({confidence:.2f})"
            
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text'], 1)
        
        return frame
    
    def _draw_faces(self, frame: np.ndarray, faces: List[Dict]) -> np.ndarray:
        """Draw face detections"""
        for face in faces:
            bbox = face.get('bbox')
            if not bbox:
                continue
            
            x1, y1, x2, y2 = bbox
            confidence = face.get('confidence', 0)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.colors['face'], 2)
            
            # Draw label
            label = f"Face {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['face'], 2)
        
        return frame
    
    def _draw_eyes(self, frame: np.ndarray, eyes: List[Dict]) -> np.ndarray:
        """Draw eye detections"""
        for eye in eyes:
            bbox = eye.get('bbox')
            if not bbox:
                continue
            
            x1, y1, x2, y2 = bbox
            side = eye.get('side', 'unknown')
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.colors['eye'], 1)
            
            # Draw label
            cv2.putText(frame, side, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, self.colors['eye'], 1)
        
        return frame
    
    def _draw_currency(self, frame: np.ndarray, currency: List[Dict]) -> np.ndarray:
        """Draw currency detections"""
        for curr in currency:
            bbox = curr.get('bbox')
            if not bbox:
                continue
            
            x1, y1, x2, y2 = bbox
            curr_type = curr.get('type', 'unknown')
            currency_code = curr.get('currency', 'unknown')
            denomination = curr.get('denomination', 0)
            confidence = curr.get('confidence', 0)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.colors['currency'], 2)
            
            # Draw label
            label = f"{currency_code} {denomination} ({curr_type})"
            if self.config.get('show_confidence', True):
                label += f" {confidence:.2f}"
            
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), self.colors['currency'], -1)
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return frame
    
    def _draw_medicine(self, frame: np.ndarray, medicine: List[Dict]) -> np.ndarray:
        """Draw medicine detections"""
        for med in medicine:
            bbox = med.get('bbox')
            if not bbox:
                continue
            
            x1, y1, x2, y2 = bbox
            med_type = med.get('type', 'unknown')
            name = med.get('name', 'unknown')
            confidence = med.get('confidence', 0)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.colors['medicine'], 2)
            
            # Draw label
            label = f"{med_type}: {name}"
            if self.config.get('show_confidence', True):
                label += f" {confidence:.2f}"
            
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['medicine'], 1)
        
        return frame
    
    def _draw_food(self, frame: np.ndarray, food: List[Dict]) -> np.ndarray:
        """Draw food detections"""
        for item in food:
            bbox = item.get('bbox')
            if not bbox:
                continue
            
            x1, y1, x2, y2 = bbox
            food_name = item.get('food_name', 'unknown')
            category = item.get('category', '')
            confidence = item.get('confidence', 0)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.colors['food'], 2)
            
            # Draw label
            label = f"{food_name} ({category})"
            if self.config.get('show_confidence', True):
                label += f" {confidence:.2f}"
            
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['food'], 1)
        
        return frame
    
    def _draw_hands(self, frame: np.ndarray, hand_data: dict) -> np.ndarray:
        """Draw hand detection and gestures"""
        hands = hand_data.get('hands', [])
        cursor_pos = hand_data.get('cursor_position')
        gesture_command = hand_data.get('gesture_command')
        
        # Draw each detected hand
        for hand in hands:
            bbox = hand.get('bbox')
            if not bbox:
                continue
            
            x1, y1, x2, y2 = bbox
            gesture = hand.get('gesture', 'unknown')
            handedness = hand.get('handedness', 'Unknown')
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{handedness} - {gesture}"
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw landmarks if available
            landmarks = hand.get('landmarks', [])
            if landmarks:
                for lm in landmarks:
                    x, y = int(lm['x']), int(lm['y'])
                    cv2.circle(frame, (x, y), 3, (255, 0, 255), -1)
        
        # Draw cursor position
        if cursor_pos:
            cv2.circle(frame, cursor_pos, 10, (0, 255, 255), 2)
            cv2.circle(frame, cursor_pos, 3, (0, 255, 255), -1)
            
            # Draw crosshair
            cv2.line(frame, (cursor_pos[0] - 15, cursor_pos[1]), 
                    (cursor_pos[0] + 15, cursor_pos[1]), (0, 255, 255), 2)
            cv2.line(frame, (cursor_pos[0], cursor_pos[1] - 15), 
                    (cursor_pos[0], cursor_pos[1] + 15), (0, 255, 255), 2)
        
        # Draw gesture command
        if gesture_command:
            cv2.putText(frame, f"Command: {gesture_command}", (10, frame.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return frame
    
    def _draw_summary(self, frame: np.ndarray, summary: Dict, fps: float, 
                     gesture_mode: str = 'detection', hands_data: dict = None) -> np.ndarray:
        """Draw summary information"""
        h, w = frame.shape[:2]
        
        # Create semi-transparent overlay for info panel
        overlay = frame.copy()
        panel_height = 240 if hands_data else 200
        cv2.rectangle(overlay, (10, 10), (350, panel_height), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Draw info text
        y_offset = 30
        line_height = 20
        
        info_lines = [
            f"FPS: {fps:.1f}",
            f"Mode: {gesture_mode.upper()}",
            f"Objects: {summary.get('total_objects', 0)}",
            f"Faces: {summary.get('total_faces', 0)}",
            f"Text Regions: {summary.get('total_text_regions', 0)}",
            f"Currency: ${summary.get('currency_total', 0):.2f}",
            f"Medicine: {summary.get('total_medicine', 0)}",
            f"Food: {summary.get('total_food', 0)}"
        ]
        
        # Add hand gesture info if available
        if hands_data:
            num_hands = len(hands_data.get('hands', []))
            gesture_cmd = hands_data.get('gesture_command', 'None')
            info_lines.append(f"Hands: {num_hands} | Gesture: {gesture_cmd}")
        
        for line in info_lines:
            cv2.putText(frame, line, (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += line_height
        
        # Draw mode indicator in top-right corner
        mode_colors = {
            'detection': (0, 255, 0),
            'cursor': (255, 165, 0),
            '3d': (255, 0, 255)
        }
        mode_color = mode_colors.get(gesture_mode, (255, 255, 255))
        
        # Mode indicator box
        mode_text = f"MODE: {gesture_mode.upper()}"
        (text_w, text_h), _ = cv2.getTextSize(mode_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        mode_x = w - text_w - 30
        mode_y = 30
        
        cv2.rectangle(frame, (mode_x - 10, mode_y - text_h - 10), 
                     (mode_x + text_w + 10, mode_y + 10), mode_color, -1)
        cv2.putText(frame, mode_text, (mode_x, mode_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Draw keyboard shortcuts at bottom
        shortcuts = "Keys: Q=Quit | M=Mode | C=Cube | P=Pyramid | S=Sphere | Y=Cylinder | X=Clear"
        cv2.putText(frame, shortcuts, (10, h - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame
