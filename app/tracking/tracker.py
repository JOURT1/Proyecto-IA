"""
Multi-Object Tracking Module using ByteTrack.
Handles persistent tracking of vehicles across frames.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import deque
import cv2

from app.utils.config import Config, get_logger
from app.detection.detector import Detection


@dataclass
class Track:
    """Represents a tracked vehicle."""
    track_id: int
    class_name: str
    
    # Position history (center points)
    position_history: deque = field(default_factory=lambda: deque(maxlen=120))
    
    # Current state
    current_bbox: Tuple[int, int, int, int] = None  # x1, y1, x2, y2
    current_center: Tuple[float, float] = None
    current_confidence: float = 0.0
    
    # Temporal information
    age: int = 0  # Frames since created
    hits: int = 0  # Frames since last detection match
    time_since_last_update: int = 0
    
    # Movement properties
    velocity: Tuple[float, float] = (0.0, 0.0)
    direction_angle: float = 0.0  # degrees
    speed: float = 0.0  # pixels per frame
    
    # Statistics
    max_age: int = 30  # Maximum frames without update before removal
    
    def update(self, detection: Detection) -> None:
        """Update track with new detection."""
        self.current_bbox = detection.bbox
        self.current_center = detection.center
        self.current_confidence = detection.confidence
        self.position_history.append(detection.center)
        
        # Calculate velocity
        if len(self.position_history) >= 2:
            prev_center = list(self.position_history)[-2]
            curr_center = detection.center
            
            dx = curr_center[0] - prev_center[0]
            dy = curr_center[1] - prev_center[1]
            
            self.velocity = (dx, dy)
            self.speed = np.sqrt(dx**2 + dy**2)
            
            # Calculate direction angle (degrees)
            if self.speed > 0:
                self.direction_angle = np.degrees(np.arctan2(dy, dx))
        
        self.time_since_last_update = 0
        self.hits += 1
        self.age += 1
    
    def predict(self) -> Tuple[int, int, int, int]:
        """
        Predict next position based on velocity.
        
        Returns:
            Predicted bounding box
        """
        if self.current_bbox is None:
            return None
        
        x1, y1, x2, y2 = self.current_bbox
        dx, dy = self.velocity
        
        return (int(x1 + dx), int(y1 + dy), int(x2 + dx), int(y2 + dy))
    
    def increment_age(self) -> None:
        """Increment age and time since last update."""
        self.age += 1
        self.time_since_last_update += 1
    
    def is_confirmed(self, min_hits: int = 3) -> bool:
        """Check if track is confirmed (has minimum hits)."""
        return self.hits >= min_hits
    
    def is_stale(self) -> bool:
        """Check if track is stale (no updates for too long)."""
        return self.time_since_last_update > self.max_age
    
    def get_trajectory(self, max_length: Optional[int] = None) -> List[Tuple[float, float]]:
        """
        Get trajectory history.
        
        Args:
            max_length: Maximum number of points to return
            
        Returns:
            List of (x, y) positions
        """
        trajectory = list(self.position_history)
        
        if max_length is not None and len(trajectory) > max_length:
            trajectory = trajectory[-max_length:]
        
        return trajectory


class ByteTracker:
    """
    SimpleByteTrack implementation for multi-object tracking.
    Based on ByteTrack algorithm.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the tracker.
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
        self.logger = get_logger()
        
        # Load configuration
        self.track_thresh = self.config.get('tracking.track_thresh', 0.5)
        self.track_buffer = self.config.get('tracking.track_buffer', 30)
        self.match_thresh = self.config.get('tracking.match_thresh', 0.8)
        self.max_trajectory_length = self.config.get('tracking.trajectory_history_length', 60)
        
        # State
        self.tracks: List[Track] = []
        self.next_track_id = 1
        
        self.logger.info("ByteTracker initialized")
    
    def update(self, detections: List[Detection]) -> List[Track]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of Detection objects
            
        Returns:
            List of confirmed tracks
        """
        # Increment age of all tracks
        for track in self.tracks:
            track.increment_age()
        
        # Match detections to tracks
        matched_pairs, unmatched_dets, unmatched_tracks = self._match_detections(detections)
        
        # Update matched tracks
        for track_idx, det_idx in matched_pairs:
            self.tracks[track_idx].update(detections[det_idx])
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            self._create_track(detections[det_idx])
        
        # Remove stale tracks
        self.tracks = [t for t in self.tracks if not t.is_stale()]
        
        # Return confirmed tracks
        confirmed_tracks = [t for t in self.tracks if t.is_confirmed()]
        return confirmed_tracks
    
    def _match_detections(self, detections: List[Detection]) -> Tuple[List, List, List]:
        """
        Match detections to existing tracks.
        
        Returns:
            Tuple of (matched_pairs, unmatched_det_indices, unmatched_track_indices)
        """
        if len(self.tracks) == 0:
            unmatched_dets = list(range(len(detections)))
            return [], unmatched_dets, []
        
        if len(detections) == 0:
            unmatched_tracks = list(range(len(self.tracks)))
            return [], [], unmatched_tracks
        
        # Compute IoU matrix
        iou_matrix = self._compute_iou_matrix(self.tracks, detections)
        
        # Greedy matching
        matched_pairs = []
        used_dets = set()
        used_tracks = set()
        
        # Sort by IoU in descending order
        indices = np.argsort(iou_matrix.flatten())[::-1]
        
        for idx in indices:
            track_idx = idx // len(detections)
            det_idx = idx % len(detections)
            
            if track_idx in used_tracks or det_idx in used_dets:
                continue
            
            if iou_matrix[track_idx, det_idx] > self.match_thresh:
                matched_pairs.append((track_idx, det_idx))
                used_tracks.add(track_idx)
                used_dets.add(det_idx)
        
        unmatched_dets = [i for i in range(len(detections)) if i not in used_dets]
        unmatched_tracks = [i for i in range(len(self.tracks)) if i not in used_tracks]
        
        return matched_pairs, unmatched_dets, unmatched_tracks
    
    def _compute_iou_matrix(self, tracks: List[Track], detections: List[Detection]) -> np.ndarray:
        """
        Compute IoU matrix between tracks and detections.
        
        Args:
            tracks: List of tracks
            detections: List of detections
            
        Returns:
            IoU matrix (num_tracks x num_detections)
        """
        iou_matrix = np.zeros((len(tracks), len(detections)))
        
        for i, track in enumerate(tracks):
            if track.current_bbox is None:
                continue
            
            for j, detection in enumerate(detections):
                iou = self._compute_iou(track.current_bbox, detection.bbox)
                iou_matrix[i, j] = iou
        
        return iou_matrix
    
    def _compute_iou(self, box1: Tuple, box2: Tuple) -> float:
        """
        Compute Intersection over Union between two bounding boxes.
        
        Args:
            box1: (x1, y1, x2, y2)
            box2: (x1, y1, x2, y2)
            
        Returns:
            IoU value
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Intersection area
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        # Union area
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area
    
    def _create_track(self, detection: Detection) -> None:
        """
        Create a new track for a detection.
        
        Args:
            detection: Detection object
        """
        track = Track(
            track_id=self.next_track_id,
            class_name=detection.class_name,
            current_bbox=detection.bbox,
            current_center=detection.center,
            current_confidence=detection.confidence,
            max_age=self.track_buffer
        )
        track.position_history.append(detection.center)
        track.hits = 1
        track.age = 1
        
        self.tracks.append(track)
        self.next_track_id += 1
        
        self.logger.debug(f"New track created: ID={track.track_id}")
    
    def get_tracks(self, confirmed_only: bool = True) -> List[Track]:
        """
        Get current tracks.
        
        Args:
            confirmed_only: If True, return only confirmed tracks
            
        Returns:
            List of tracks
        """
        if confirmed_only:
            return [t for t in self.tracks if t.is_confirmed()]
        return self.tracks
    
    def reset(self) -> None:
        """Reset tracker."""
        self.tracks = []
        self.next_track_id = 1
