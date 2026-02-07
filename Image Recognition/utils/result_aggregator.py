"""
Aggregate and post-process detection results
"""

import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class ResultAggregator:
    """Aggregate results from multiple detectors"""
    
    def __init__(self):
        pass
    
    def aggregate(self, results: Dict) -> Dict:
        """
        Aggregate results from all detectors
        
        Args:
            results: Dict with keys like 'objects', 'text', 'faces', etc.
        
        Returns:
            Aggregated results with unified format
        """
        aggregated = {
            'objects': [],
            'text': [],
            'faces': [],
            'eyes': [],
            'currency': [],
            'medicine': [],
            'food': [],
            'depth_map': None,
            'summary': {}
        }
        
        # Process object detections
        if results.get('objects'):
            aggregated['objects'] = results['objects']
        
        # Process text detections
        if results.get('text'):
            aggregated['text'] = results['text']
        
        # Process face/eye detections
        if results.get('faces'):
            face_data = results['faces']
            aggregated['faces'] = face_data.get('faces', [])
            aggregated['eyes'] = face_data.get('eyes', [])
        
        # Process currency detections
        if results.get('currency'):
            aggregated['currency'] = results['currency']
        
        # Process medicine detections
        if results.get('medicine'):
            aggregated['medicine'] = results['medicine']
        
        # Process food detections
        if results.get('food'):
            aggregated['food'] = results['food']
        
        # Process depth map
        if results.get('depth') is not None:
            aggregated['depth_map'] = results['depth']
            
            # Add distance information to objects
            aggregated['objects'] = self._add_distance_to_objects(
                aggregated['objects'], 
                aggregated['depth_map']
            )
        
        # Generate summary
        aggregated['summary'] = self._generate_summary(aggregated)
        
        return aggregated
    
    def _add_distance_to_objects(self, objects: List[Dict], depth_map: np.ndarray) -> List[Dict]:
        """Add distance information to object detections"""
        if depth_map is None:
            return objects
        
        for obj in objects:
            bbox = obj.get('bbox')
            if bbox:
                x1, y1, x2, y2 = bbox
                
                # Get depth in bounding box
                roi_depth = depth_map[y1:y2, x1:x2]
                
                if roi_depth.size > 0:
                    # Use median depth
                    median_depth = np.median(roi_depth)
                    
                    # Convert to approximate distance (0-10 meters)
                    distance = (1.0 - median_depth) * 10.0
                    obj['distance'] = float(distance)
        
        return objects
    
    def _generate_summary(self, results: Dict) -> Dict:
        """Generate summary statistics"""
        summary = {
            'total_objects': len(results['objects']),
            'total_text_regions': len(results['text']),
            'total_faces': len(results['faces']),
            'total_eyes': len(results['eyes']),
            'total_currency': len(results['currency']),
            'total_medicine': len(results['medicine']),
            'total_food': len(results['food']),
            'object_classes': {},
            'currency_total': 0.0
        }
        
        # Count object classes
        for obj in results['objects']:
            class_name = obj.get('class', 'unknown')
            summary['object_classes'][class_name] = summary['object_classes'].get(class_name, 0) + 1
        
        # Calculate total currency value
        for curr in results['currency']:
            if curr.get('denomination'):
                summary['currency_total'] += curr['denomination']
        
        return summary
