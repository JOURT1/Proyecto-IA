"""
Visualization Module.
Handles drawing detections, tracks, and alerts on video frames.
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from app.utils.config import Config, get_logger
from app.tracking.tracker import Track
from app.collision.detector import CollisionEvent
from app.severity.classifier import SeverityAssessment, SeverityLevel


class FrameVisualizer:
    """
    Draws detections, tracks, and events on video frames.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize visualizer.
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
        self.logger = get_logger()
        
        # Load visualization configuration
        self.bbox_color = tuple(self.config.get('visualization.bbox_color', [0, 255, 0]))
        self.bbox_thickness = self.config.get('visualization.bbox_thickness', 2)
        
        self.trajectory_color = tuple(self.config.get('visualization.trajectory_color', [255, 0, 0]))
        self.trajectory_thickness = self.config.get('visualization.trajectory_thickness', 1)
        self.trajectory_max_age = self.config.get('visualization.trajectory_max_age', 30)
        
        self.text_size = self.config.get('visualization.text_size', 0.5)
        self.text_thickness = self.config.get('visualization.text_thickness', 1)
        self.text_color = tuple(self.config.get('visualization.text_color', [255, 255, 255]))
        
        self.alert_text_size = self.config.get('visualization.alert_text_size', 1.2)
        self.alert_text_thickness = self.config.get('visualization.alert_text_thickness', 2)
        
        self.severity_colors = self.config.get('visualization.severity_colors', {
            'leve': [0, 255, 255],
            'moderado': [0, 165, 255],
            'severo': [0, 0, 255]
        })
        
        # Convert to tuples
        self.severity_colors = {
            k: tuple(v) for k, v in self.severity_colors.items()
        }
        
        self.draw_progress_bar = self.config.get('visualization.draw_progress_bar', True)
        self.progress_bar_height = self.config.get('visualization.progress_bar_height', 30)
    
    def draw_frame(
        self,
        frame: np.ndarray,
        tracks: List[Track],
        collisions: List[CollisionEvent],
        severity_assessments: Optional[Dict[Tuple, SeverityAssessment]] = None,
        frame_number: int = 0,
        total_frames: int = 0
    ) -> np.ndarray:
        """
        Draw all visualization elements on frame.
        
        Args:
            frame: Input frame
            tracks: List of tracked vehicles
            collisions: List of collision events
            severity_assessments: Dict mapping (track_ids) tuple to SeverityAssessment
            frame_number: Current frame number
            total_frames: Total number of frames in video
            
        Returns:
            Annotated frame
        """
        output_frame = frame.copy()
        
        # Draw trajectories first (background)
        output_frame = self._draw_trajectories(output_frame, tracks)
        
        # Draw bounding boxes and IDs
        output_frame = self._draw_tracks(output_frame, tracks)
        
        # Draw collision alerts
        if collisions:
            output_frame = self._draw_collisions(
                output_frame, collisions, severity_assessments
            )
        
        # Draw progress bar
        if self.draw_progress_bar and total_frames > 0:
            output_frame = self._draw_progress_bar(
                output_frame, frame_number, total_frames
            )
        
        # Draw timestamp
        output_frame = self._draw_timestamp(output_frame, frame_number)
        
        return output_frame
    
    def _draw_trajectories(self, frame: np.ndarray, tracks: List[Track]) -> np.ndarray:
        """Draw vehicle trajectories."""
        output_frame = frame.copy()
        
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            trajectory = track.get_trajectory(self.trajectory_max_age)
            
            if len(trajectory) < 2:
                continue
            
            # Draw trajectory line
            for i in range(len(trajectory) - 1):
                pt1 = tuple(map(int, trajectory[i]))
                pt2 = tuple(map(int, trajectory[i + 1]))
                
                cv2.line(
                    output_frame, pt1, pt2,
                    self.trajectory_color,
                    self.trajectory_thickness
                )
        
        return output_frame
    
    def _draw_tracks(self, frame: np.ndarray, tracks: List[Track]) -> np.ndarray:
        """Draw track bounding boxes and IDs."""
        output_frame = frame.copy()
        
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            if track.current_bbox is None:
                continue
            
            x1, y1, x2, y2 = track.current_bbox
            
            # Draw bounding box
            cv2.rectangle(
                output_frame, (x1, y1), (x2, y2),
                self.bbox_color,
                self.bbox_thickness
            )
            
            # Draw ID
            label = f"ID: {track.track_id}"
            
            if track.speed is not None:
                label += f" v:{track.speed:.1f}px"
            
            text_size = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, self.text_size, self.text_thickness
            )[0]
            
            # Background for text
            cv2.rectangle(
                output_frame,
                (x1, y1 - text_size[1] - 5),
                (x1 + text_size[0] + 5, y1),
                self.bbox_color,
                -1
            )
            
            # Text
            cv2.putText(
                output_frame, label,
                (x1 + 2, y1 - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.text_size,
                self.text_color,
                self.text_thickness
            )
        
        return output_frame
    
    def _draw_collisions(
        self,
        frame: np.ndarray,
        collisions: List[CollisionEvent],
        severity_assessments: Optional[Dict[Tuple, SeverityAssessment]] = None
    ) -> np.ndarray:
        """Draw collision alerts and information."""
        output_frame = frame.copy()
        
        h, w = frame.shape[:2]
        
        for i, collision in enumerate(collisions):
            # Get severity assessment
            severity = None
            if severity_assessments:
                key = tuple(sorted(collision.track_ids))
                severity = severity_assessments.get(key)
            
            # Get color based on severity
            if severity:
                level = severity.level
                if level == SeverityLevel.SEVERO:
                    alert_color = self.severity_colors['severo']
                    alert_text = "ALERTA SEVERA"
                elif level == SeverityLevel.MODERADO:
                    alert_color = self.severity_colors['moderado']
                    alert_text = "ALERTA MODERADA"
                else:
                    alert_color = self.severity_colors['leve']
                    alert_text = "ALERTA LEVE"
            else:
                alert_color = (0, 0, 255)  # Red
                alert_text = "COLISIÓN DETECTADA"
            
            # Draw alert box
            alert_y = 50 + i * 60
            
            cv2.rectangle(
                output_frame,
                (10, alert_y - 40),
                (w - 10, alert_y),
                alert_color,
                -1
            )
            
            # Alert text
            cv2.putText(
                output_frame, alert_text,
                (20, alert_y - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.alert_text_size,
                (255, 255, 255),
                self.alert_text_thickness
            )
            
            # Details text
            if severity:
                details = f"Vehículos: {collision.track_ids} | Confianza: {collision.confidence:.2f} | Puntuación: {severity.score:.1f}"
            else:
                details = f"Vehículos: {collision.track_ids} | Confianza: {collision.confidence:.2f}"
            
            cv2.putText(
                output_frame, details,
                (20, alert_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
        
        return output_frame
    
    def _draw_progress_bar(
        self,
        frame: np.ndarray,
        frame_number: int,
        total_frames: int
    ) -> np.ndarray:
        """Draw progress bar at bottom of frame."""
        output_frame = frame.copy()
        
        h, w = frame.shape[:2]
        
        # Calculate progress
        if total_frames > 0:
            progress = frame_number / total_frames
        else:
            progress = 0.0
        
        progress = max(0.0, min(1.0, progress))
        
        # Draw background bar
        cv2.rectangle(
            output_frame,
            (0, h - self.progress_bar_height),
            (w, h),
            (50, 50, 50),
            -1
        )
        
        # Draw progress bar
        progress_width = int(w * progress)
        cv2.rectangle(
            output_frame,
            (0, h - self.progress_bar_height),
            (progress_width, h),
            (0, 255, 0),
            -1
        )
        
        # Draw text
        progress_text = f"{frame_number}/{total_frames} | {progress*100:.1f}%"
        cv2.putText(
            output_frame, progress_text,
            (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )
        
        return output_frame
    
    def _draw_timestamp(self, frame: np.ndarray, frame_number: int) -> np.ndarray:
        """Draw timestamp on frame."""
        output_frame = frame.copy()
        
        fps = self.config.get('kinematics.fps', 30)
        seconds = frame_number / fps
        
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        
        timestamp_text = f"[{minutes:02d}:{secs:02d}]"
        
        cv2.putText(
            output_frame, timestamp_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        return output_frame
