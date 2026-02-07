"""
High-performance object detection with multiple model fallbacks
Enhanced with cloud API support for person detection
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class ObjectDetector:
    """Multi-model object detector with fallback mechanisms and cloud support"""
    
    def __init__(self, config: Dict, optimizer):
        self.config = config
        self.optimizer = optimizer
        self.models = {}
        self.active_model = None
        
        # Cloud detection for enhanced accuracy
        self.use_cloud = config.get('use_cloud_detection', True)
        self.cloud_detector = None
        self.person_detection_mode = config.get('person_detection_mode', 'hybrid')
        
        self._init_cloud_detector()
        
        # Initialize primary model
        self._load_primary_model()
        
        # Preload backup models if needed
        if self.config.get('preload_backups', False):
            self._load_backup_models()
    
    def _init_cloud_detector(self):
        """Initialize cloud vision detector for enhanced accuracy"""
        if self.use_cloud:
            try:
                from detectors.cloud_vision_detector import CloudVisionDetector
                cloud_config = self.config.get('cloud_apis', {})
                self.cloud_detector = CloudVisionDetector(cloud_config)
                
                if self.cloud_detector.is_available():
                    logger.info("Cloud detection enabled for object detector")
                else:
                    self.cloud_detector = None
                    
            except Exception as e:
                logger.warning(f"Cloud detector init failed: {e}")
                self.cloud_detector = None
    
    def _load_primary_model(self):
        """Load primary detection model (YOLOv8)"""
        try:
            model_name = self.config['primary']
            logger.info(f"Loading primary model: {model_name}")
            
            # Load YOLOv8
            self.models['yolov8'] = YOLO(f'{model_name}.pt')
            
            # Optimize for GPU if available
            if torch.cuda.is_available() and self.optimizer.use_gpu:
                self.models['yolov8'].to('cuda')
                logger.info("YOLOv8 loaded on GPU")
            else:
                logger.info("YOLOv8 loaded on CPU")
            
            self.active_model = 'yolov8'
            
        except Exception as e:
            logger.error(f"Failed to load primary model: {e}")
            self._load_backup_models()
    
    def _load_backup_models(self):
        """Load backup detection models"""
        for backup in self.config.get('backup', []):
            try:
                if backup == 'detectron2':
                    self._load_detectron2()
                elif backup == 'faster_rcnn':
                    self._load_faster_rcnn()
            except Exception as e:
                logger.warning(f"Failed to load backup {backup}: {e}")
    
    def _load_detectron2(self):
        """Load Detectron2 model"""
        try:
            from detectron2.engine import DefaultPredictor
            from detectron2.config import get_cfg
            from detectron2 import model_zoo
            
            cfg = get_cfg()
            cfg.merge_from_file(model_zoo.get_config_file(
                "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
            ))
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.config['confidence_threshold']
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
            )
            
            if torch.cuda.is_available() and self.optimizer.use_gpu:
                cfg.MODEL.DEVICE = 'cuda'
            else:
                cfg.MODEL.DEVICE = 'cpu'
            
            self.models['detectron2'] = DefaultPredictor(cfg)
            logger.info("Detectron2 loaded successfully")
            
            if self.active_model is None:
                self.active_model = 'detectron2'
                
        except Exception as e:
            logger.warning(f"Detectron2 loading failed: {e}")
    
    def _load_faster_rcnn(self):
        """Load Faster R-CNN from torchvision"""
        try:
            import torchvision
            from torchvision.models.detection import fasterrcnn_resnet50_fpn
            
            model = fasterrcnn_resnet50_fpn(pretrained=True)
            model.eval()
            
            if torch.cuda.is_available() and self.optimizer.use_gpu:
                model = model.cuda()
            
            self.models['faster_rcnn'] = model
            logger.info("Faster R-CNN loaded successfully")
            
            if self.active_model is None:
                self.active_model = 'faster_rcnn'
                
        except Exception as e:
            logger.warning(f"Faster R-CNN loading failed: {e}")
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect objects in frame
        Returns list of detections with format:
        [{
            'class': str,
            'confidence': float,
            'bbox': [x1, y1, x2, y2],
            'center': [cx, cy]
        }]
        """
        if self.active_model is None:
            return []
        
        try:
            if self.active_model == 'yolov8':
                return self._detect_yolov8(frame)
            elif self.active_model == 'detectron2':
                return self._detect_detectron2(frame)
            elif self.active_model == 'faster_rcnn':
                return self._detect_faster_rcnn(frame)
        except Exception as e:
            logger.warning(f"Detection failed with {self.active_model}: {e}")
            # Try fallback
            return self._detect_with_fallback(frame)
        
        return []
    
    def detect_people(self, frame: np.ndarray) -> List[Dict]:
        """
        Optimized person detection using best available method
        Supports modes: 'local', 'cloud', 'hybrid'
        """
        if self.person_detection_mode == 'cloud' and self.cloud_detector:
            return self._detect_people_cloud(frame)
        elif self.person_detection_mode == 'hybrid' and self.cloud_detector:
            return self._detect_people_hybrid(frame)
        else:
            return self._detect_people_local(frame)
    
    def _detect_people_local(self, frame: np.ndarray) -> List[Dict]:
        """Detect people using local YOLO model"""
        all_detections = self.detect(frame)
        return [d for d in all_detections if d.get('class', '').lower() == 'person']
    
    def _detect_people_cloud(self, frame: np.ndarray) -> List[Dict]:
        """Detect people using cloud API"""
        try:
            cloud_results = self.cloud_detector.detect_people(frame)
            if cloud_results:
                return cloud_results
        except Exception as e:
            logger.warning(f"Cloud person detection failed: {e}")
        
        # Fallback to local
        return self._detect_people_local(frame)
    
    def _detect_people_hybrid(self, frame: np.ndarray) -> List[Dict]:
        """
        Hybrid detection: Use local YOLO for speed, verify with cloud for accuracy
        """
        # First, get local detections (fast)
        local_people = self._detect_people_local(frame)
        
        if not local_people:
            # If no local detections, try cloud
            return self._detect_people_cloud(frame)
        
        # Optionally verify high-importance detections with cloud
        if self.cloud_detector and len(local_people) > 0:
            try:
                cloud_people = self.cloud_detector.detect_people(frame)
                
                if cloud_people:
                    # Merge results, preferring cloud for confidence
                    merged = self._merge_detections(local_people, cloud_people)
                    return merged
            except Exception as e:
                logger.debug(f"Cloud verification skipped: {e}")
        
        return local_people
    
    def _merge_detections(self, local: List[Dict], cloud: List[Dict]) -> List[Dict]:
        """Merge local and cloud detections, removing duplicates"""
        merged = []
        used_cloud = set()
        
        for local_det in local:
            best_match = None
            best_iou = 0
            
            for i, cloud_det in enumerate(cloud):
                if i in used_cloud:
                    continue
                    
                iou = self._calculate_iou(local_det['bbox'], cloud_det['bbox'])
                if iou > 0.5 and iou > best_iou:
                    best_iou = iou
                    best_match = i
            
            if best_match is not None:
                # Use cloud confidence (usually more accurate)
                merged_det = local_det.copy()
                merged_det['confidence'] = max(
                    local_det['confidence'],
                    cloud[best_match]['confidence']
                )
                merged_det['verified'] = True
                merged.append(merged_det)
                used_cloud.add(best_match)
            else:
                merged.append(local_det)
        
        # Add any cloud detections not matched
        for i, cloud_det in enumerate(cloud):
            if i not in used_cloud:
                cloud_det['source'] = 'cloud'
                merged.append(cloud_det)
        
        return merged
    
    def _calculate_iou(self, box1: List, box2: List) -> float:
        """Calculate Intersection over Union"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def _detect_yolov8(self, frame: np.ndarray) -> List[Dict]:
        """Detect using YOLOv8"""
        model = self.models['yolov8']
        
        # Run inference
        results = model(frame, conf=self.config['confidence_threshold'], 
                       iou=self.config['nms_threshold'], verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = model.names[cls]
                
                detections.append({
                    'class': class_name,
                    'confidence': conf,
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'center': [int((x1 + x2) / 2), int((y1 + y2) / 2)]
                })
        
        return detections
    
    def _detect_detectron2(self, frame: np.ndarray) -> List[Dict]:
        """Detect using Detectron2"""
        from detectron2.data import MetadataCatalog
        
        predictor = self.models['detectron2']
        outputs = predictor(frame)
        
        instances = outputs["instances"].to("cpu")
        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()
        classes = instances.pred_classes.numpy()
        
        metadata = MetadataCatalog.get(predictor.cfg.DATASETS.TRAIN[0])
        class_names = metadata.thing_classes
        
        detections = []
        for box, score, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = box
            detections.append({
                'class': class_names[cls],
                'confidence': float(score),
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'center': [int((x1 + x2) / 2), int((y1 + y2) / 2)]
            })
        
        return detections
    
    def _detect_faster_rcnn(self, frame: np.ndarray) -> List[Dict]:
        """Detect using Faster R-CNN"""
        import torchvision.transforms as T
        
        model = self.models['faster_rcnn']
        
        # Prepare image
        transform = T.Compose([T.ToTensor()])
        img_tensor = transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if torch.cuda.is_available() and self.optimizer.use_gpu:
            img_tensor = img_tensor.cuda()
        
        # Inference
        with torch.no_grad():
            predictions = model([img_tensor])[0]
        
        # COCO class names
        COCO_CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                       'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                       'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                       'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                       'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                       'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                       'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                       'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                       'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                       'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                       'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                       'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                       'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                       'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        
        detections = []
        boxes = predictions['boxes'].cpu().numpy()
        scores = predictions['scores'].cpu().numpy()
        labels = predictions['labels'].cpu().numpy()
        
        for box, score, label in zip(boxes, scores, labels):
            if score >= self.config['confidence_threshold']:
                x1, y1, x2, y2 = box
                detections.append({
                    'class': COCO_CLASSES[label - 1] if label <= len(COCO_CLASSES) else 'unknown',
                    'confidence': float(score),
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'center': [int((x1 + x2) / 2), int((y1 + y2) / 2)]
                })
        
        return detections
    
    def _detect_with_fallback(self, frame: np.ndarray) -> List[Dict]:
        """Try detection with backup models"""
        for model_name in ['detectron2', 'faster_rcnn']:
            if model_name in self.models and model_name != self.active_model:
                try:
                    logger.info(f"Trying fallback model: {model_name}")
                    self.active_model = model_name
                    return self.detect(frame)
                except Exception as e:
                    logger.warning(f"Fallback {model_name} failed: {e}")
        
        return []
