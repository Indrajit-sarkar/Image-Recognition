"""
3D structure creation using hand gestures
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)


class Structure3DCreator:
    """Create and manipulate 3D structures with hand gestures"""
    
    def __init__(self):
        self.structures = []
        self.current_structure = None
        self.drawing_mode = False
        self.rotation = [0, 0, 0]  # x, y, z rotation
        self.scale = 1.0
        self.position = [0, 0, 0]
        
        # Drawing state
        self.points = []
        self.edges = []
        
        # Predefined 3D shapes
        self.shapes = {
            'cube': self._create_cube(),
            'pyramid': self._create_pyramid(),
            'sphere': self._create_sphere(),
            'cylinder': self._create_cylinder()
        }
    
    def _create_cube(self) -> Dict:
        """Create a cube structure"""
        size = 100
        vertices = np.array([
            [-size, -size, -size],
            [size, -size, -size],
            [size, size, -size],
            [-size, size, -size],
            [-size, -size, size],
            [size, -size, size],
            [size, size, size],
            [-size, size, size]
        ], dtype=float)
        
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Back face
            (4, 5), (5, 6), (6, 7), (7, 4),  # Front face
            (0, 4), (1, 5), (2, 6), (3, 7)   # Connecting edges
        ]
        
        return {'vertices': vertices, 'edges': edges, 'type': 'cube'}
    
    def _create_pyramid(self) -> Dict:
        """Create a pyramid structure"""
        size = 100
        vertices = np.array([
            [-size, -size, 0],
            [size, -size, 0],
            [size, size, 0],
            [-size, size, 0],
            [0, 0, size * 1.5]  # Apex
        ], dtype=float)
        
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Base
            (0, 4), (1, 4), (2, 4), (3, 4)   # Sides to apex
        ]
        
        return {'vertices': vertices, 'edges': edges, 'type': 'pyramid'}
    
    def _create_sphere(self) -> Dict:
        """Create a sphere structure (approximated with vertices)"""
        radius = 100
        vertices = []
        edges = []
        
        # Create sphere using latitude and longitude
        lat_steps = 10
        lon_steps = 10
        
        for i in range(lat_steps + 1):
            lat = np.pi * i / lat_steps - np.pi / 2
            for j in range(lon_steps):
                lon = 2 * np.pi * j / lon_steps
                
                x = radius * np.cos(lat) * np.cos(lon)
                y = radius * np.cos(lat) * np.sin(lon)
                z = radius * np.sin(lat)
                
                vertices.append([x, y, z])
        
        vertices = np.array(vertices, dtype=float)
        
        # Create edges
        for i in range(lat_steps):
            for j in range(lon_steps):
                current = i * lon_steps + j
                next_lon = i * lon_steps + (j + 1) % lon_steps
                next_lat = (i + 1) * lon_steps + j
                
                edges.append((current, next_lon))
                if i < lat_steps:
                    edges.append((current, next_lat))
        
        return {'vertices': vertices, 'edges': edges, 'type': 'sphere'}
    
    def _create_cylinder(self) -> Dict:
        """Create a cylinder structure"""
        radius = 80
        height = 150
        segments = 12
        
        vertices = []
        edges = []
        
        # Bottom circle
        for i in range(segments):
            angle = 2 * np.pi * i / segments
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            vertices.append([x, y, -height/2])
        
        # Top circle
        for i in range(segments):
            angle = 2 * np.pi * i / segments
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            vertices.append([x, y, height/2])
        
        vertices = np.array(vertices, dtype=float)
        
        # Create edges
        for i in range(segments):
            next_i = (i + 1) % segments
            # Bottom circle
            edges.append((i, next_i))
            # Top circle
            edges.append((i + segments, next_i + segments))
            # Vertical edges
            edges.append((i, i + segments))
        
        return {'vertices': vertices, 'edges': edges, 'type': 'cylinder'}
    
    def create_shape(self, shape_type: str) -> bool:
        """Create a new 3D shape"""
        if shape_type in self.shapes:
            self.current_structure = self.shapes[shape_type].copy()
            self.structures.append(self.current_structure)
            logger.info(f"Created {shape_type}")
            return True
        return False
    
    def rotate(self, axis: str, angle: float):
        """Rotate current structure"""
        if self.current_structure is None:
            return
        
        vertices = self.current_structure['vertices']
        angle_rad = np.radians(angle)
        
        if axis == 'x':
            rotation_matrix = np.array([
                [1, 0, 0],
                [0, np.cos(angle_rad), -np.sin(angle_rad)],
                [0, np.sin(angle_rad), np.cos(angle_rad)]
            ])
        elif axis == 'y':
            rotation_matrix = np.array([
                [np.cos(angle_rad), 0, np.sin(angle_rad)],
                [0, 1, 0],
                [-np.sin(angle_rad), 0, np.cos(angle_rad)]
            ])
        elif axis == 'z':
            rotation_matrix = np.array([
                [np.cos(angle_rad), -np.sin(angle_rad), 0],
                [np.sin(angle_rad), np.cos(angle_rad), 0],
                [0, 0, 1]
            ])
        else:
            return
        
        # Apply rotation
        self.current_structure['vertices'] = vertices @ rotation_matrix.T
    
    def scale_structure(self, factor: float):
        """Scale current structure"""
        if self.current_structure is None:
            return
        
        self.current_structure['vertices'] *= factor
        self.scale *= factor
    
    def translate(self, dx: float, dy: float, dz: float):
        """Move current structure"""
        if self.current_structure is None:
            return
        
        self.current_structure['vertices'] += np.array([dx, dy, dz])
    
    def project_to_2d(self, vertices: np.ndarray, 
                      width: int, height: int) -> np.ndarray:
        """Project 3D vertices to 2D screen coordinates"""
        # Simple perspective projection
        focal_length = 500
        
        projected = []
        for vertex in vertices:
            x, y, z = vertex
            
            # Perspective division
            if z + focal_length != 0:
                factor = focal_length / (z + focal_length)
                x_proj = int(x * factor + width / 2)
                y_proj = int(y * factor + height / 2)
            else:
                x_proj = int(x + width / 2)
                y_proj = int(y + height / 2)
            
            projected.append([x_proj, y_proj])
        
        return np.array(projected)
    
    def render(self, frame: np.ndarray) -> np.ndarray:
        """Render all 3D structures on frame"""
        if not self.structures:
            return frame
        
        h, w = frame.shape[:2]
        output = frame.copy()
        
        for structure in self.structures:
            vertices = structure['vertices']
            edges = structure['edges']
            
            # Project to 2D
            projected = self.project_to_2d(vertices, w, h)
            
            # Draw edges
            for edge in edges:
                pt1 = tuple(projected[edge[0]])
                pt2 = tuple(projected[edge[1]])
                
                # Check if points are within frame
                if (0 <= pt1[0] < w and 0 <= pt1[1] < h and
                    0 <= pt2[0] < w and 0 <= pt2[1] < h):
                    cv2.line(output, pt1, pt2, (0, 255, 255), 2)
            
            # Draw vertices
            for point in projected:
                pt = tuple(point)
                if 0 <= pt[0] < w and 0 <= pt[1] < h:
                    cv2.circle(output, pt, 4, (255, 0, 255), -1)
        
        return output
    
    def handle_gesture(self, gesture_command: str, hand_position: Tuple[int, int]):
        """Handle gesture commands for 3D manipulation"""
        if gesture_command == 'grab':
            # Start manipulation
            self.drawing_mode = True
        elif gesture_command == 'release':
            # End manipulation
            self.drawing_mode = False
        elif gesture_command == 'scroll':
            # Rotate structure
            if self.current_structure:
                self.rotate('y', 5)
    
    def clear_all(self):
        """Clear all structures"""
        self.structures = []
        self.current_structure = None
        logger.info("Cleared all 3D structures")
