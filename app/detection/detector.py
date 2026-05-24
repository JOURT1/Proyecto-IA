"""
Vehicle Detection Module using YOLO.
Handles object detection for traffic surveillance videos.
"""

import cv2
import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from ultralytics import YOLO

from app.utils.config import Config, get_logger


@dataclass
class Detection:
    """Represents a single detection."""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[float, float]  # x, y
    area: float


class VehicleDetector:
    """
    YOLO-based vehicle detector for traffic surveillance.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the detector.
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
        self.logger = get_logger()
        
        # Load configuration
        self.model_name = self.config.get('model.yolo_model', 'yolov8n.pt')
        self.device = self._get_device()
        self.confidence_threshold = self.config.get('model.confidence_threshold', 0.5)
        self.iou_threshold = self.config.get('model.iou_threshold', 0.45)
        self.vehicle_classes = self.config.get('model.vehicle_classes', 
                                              ['car', 'motorcycle', 'bus', 'truck'])
        
        # Load YOLO model
        self.logger.info(f"Loading YOLO model: {self.model_name}")
        self.model = YOLO(self.model_name)
        self.model.to(self.device)
        self.logger.info(f"Model loaded on device: {self.device}")
        
        # Get class names from model
        self.class_names = self.model.names
        self.logger.info(f"Available classes: {self.class_names}")
    
    def _get_device(self) -> str:
        """
        Determine and return the device to use.
        
        Returns:
            'cuda' or 'cpu'
        """
        device_config = self.config.get('model.yolo_device', 'auto')
        
        if device_config == 'auto':
            return '0' if torch.cuda.is_available() else 'cpu'  # 0 for first GPU
        elif device_config == 'cuda':
            return '0'
        else:
            return 'cpu'
    
    def detect(self, frame: np.ndarray, return_numpy: bool = False) -> List[Detection]:
        """
        Detect vehicles in a frame.
        
        Args:
            frame: Input frame (BGR format)
            return_numpy: If True, also returns numpy array results
            
        Returns:
            List of Detection objects
        """
        # Run inference
        results = self.model(frame, conf=self.confidence_threshold, iou=self.iou_threshold)
        
        detections = []
        
        if results and len(results) > 0:
            result = results[0]
            
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0].item())
                    class_name = self.class_names.get(class_id, f"Unknown_{class_id}")
                    
                    # Filter by vehicle classes
                    if class_name.lower() not in self.vehicle_classes:
                        continue
                    
                    confidence = float(box.conf[0].item())
                    
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Clamp to frame bounds
                    h, w = frame.shape[:2]
                    x1 = max(0, min(x1, w))
                    y1 = max(0, min(y1, h))
                    x2 = max(0, min(x2, w))
                    y2 = max(0, min(y2, h))
                    
                    # Calculate center point
                    center_x = (x1 + x2) / 2.0
                    center_y = (y1 + y2) / 2.0
                    
                    # Calculate area
                    area = (x2 - x1) * (y2 - y1)
                    
                    detection = Detection(
                        class_id=class_id,
                        class_name=class_name,
                        confidence=confidence,
                        bbox=(x1, y1, x2, y2),
                        center=(center_x, center_y),
                        area=area
                    )
                    detections.append(detection)
        
        return detections
    
    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """
        Detect vehicles in multiple frames.
        
        Args:
            frames: List of input frames
            
        Returns:
            List of lists of Detection objects
        """
        batch_detections = []
        
        for frame in frames:
            detections = self.detect(frame)
            batch_detections.append(detections)
        
        return batch_detections
    
    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        draw_confidence: bool = True
    ) -> np.ndarray:
        """
        Draw detections on the frame.
        
        Args:
            frame: Input frame
            detections: List of detections
            draw_confidence: Whether to draw confidence scores
            
        Returns:
            Frame with drawn detections
        """
        output_frame = frame.copy()
        
        config = Config()
        bbox_color = tuple(config.get('visualization.bbox_color', [0, 255, 0]))
        bbox_thickness = config.get('visualization.bbox_thickness', 2)
        text_size = config.get('visualization.text_size', 0.5)
        text_thickness = config.get('visualization.text_thickness', 1)
        text_color = tuple(config.get('visualization.text_color', [255, 255, 255]))
        
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            
            # Draw bounding box
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), bbox_color, bbox_thickness)
            
            # Draw label
            label = detection.class_name
            if draw_confidence:
                label += f" {detection.confidence:.2f}"
            
            text_size_obj = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, text_size, text_thickness)
            text_width, text_height = text_size_obj[0]
            
            # Draw text background
            cv2.rectangle(
                output_frame,
                (x1, y1 - text_height - 5),
                (x1 + text_width + 5, y1),
                bbox_color,
                -1
            )
            
            # Draw text
            cv2.putText(
                output_frame,
                label,
                (x1 + 2, y1 - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                text_size,
                text_color,
                text_thickness
            )
        
        return output_frame
    
    def set_confidence_threshold(self, threshold: float) -> None:
        """
        Update confidence threshold.
        
        Args:
            threshold: New confidence threshold (0.0-1.0)
        """
        self.confidence_threshold = max(0.0, min(1.0, threshold))
        self.logger.info(f"Confidence threshold updated to: {self.confidence_threshold}")
    
    def set_iou_threshold(self, threshold: float) -> None:
        """
        Update IoU threshold.
        
        Args:
            threshold: New IoU threshold (0.0-1.0)
        """
        self.iou_threshold = max(0.0, min(1.0, threshold))
        self.logger.info(f"IoU threshold updated to: {self.iou_threshold}")
