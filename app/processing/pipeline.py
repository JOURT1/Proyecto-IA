"""
Video Processing Pipeline.
Main pipeline that integrates detection, tracking, collision, and severity modules.
"""

import cv2
import time
from typing import List, Optional, Dict, Tuple
from pathlib import Path
import json
import csv

from app.utils.config import Config, get_logger, PerformanceMetrics, format_timestamp
from app.detection.detector import VehicleDetector
from app.tracking.tracker import ByteTracker
from app.kinematics.calculator import KinematicsCalculator
from app.collision.detector import CollisionDetector, CollisionEvent
from app.severity.classifier import SeverityClassifier, SeverityAssessment
from app.visualization.visualizer import FrameVisualizer


class VideoProcessor:
    """
    Main video processing pipeline for accident detection.
    """
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """
        Initialize the processor.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = Config()
        self.config.load_config(config_path)
        self.logger = get_logger()
        
        # Initialize components
        self.detector = VehicleDetector(self.config)
        self.tracker = ByteTracker(self.config)
        self.kinematics = KinematicsCalculator(self.config)
        self.collision_detector = CollisionDetector(self.config, self.kinematics)
        self.severity_classifier = SeverityClassifier(self.config)
        self.visualizer = FrameVisualizer(self.config)
        
        # State
        self.metrics = PerformanceMetrics()
        self.collision_history: Dict[Tuple, List[CollisionEvent]] = {}
        self.severity_history: Dict[Tuple, List[SeverityAssessment]] = {}
        
        self.logger.info("VideoProcessor initialized")
    
    def process_video(
        self,
        video_path: str,
        output_video_path: Optional[str] = None,
        visualize: bool = True,
        export_json: bool = False,
        export_csv: bool = False,
        progress_callback=None,
        frame_callback=None
    ) -> Dict:
        """
        Process a video file for accident detection.
        
        Args:
            video_path: Path to input video
            output_video_path: Path for output video (optional)
            visualize: Whether to draw annotations
            export_json: Whether to export results as JSON
            export_csv: Whether to export events as CSV
            
        Returns:
            Results dictionary with statistics and events
        """
        # Check if video exists
        if not Path(video_path).exists():
            self.logger.error(f"Video file not found: {video_path}")
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        self.logger.info(f"Processing video: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            self.logger.error(f"Failed to open video: {video_path}")
            raise RuntimeError(f"Failed to open video: {video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.logger.info(f"Video properties: {width}x{height} @ {fps} FPS, {total_frames} frames")
        
        # Setup output video writer if needed
        video_writer = None
        if output_video_path and visualize:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            if not video_writer.isOpened():
                self.logger.warning(f"Failed to open output video writer: {output_video_path}")
                video_writer = None
        
        # Reset metrics
        self.metrics.reset()
        self.metrics.start_time = time.time()
        
        # Processing loop
        frame_idx = 0
        detected_accidents = []
        
        self.logger.info("Starting video processing...")
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Process frame
                frame_start_time = time.time()
                
                # Detection
                det_start = time.time()
                detections = self.detector.detect(frame)
                self.metrics.detection_times.append(time.time() - det_start)
                
                # Tracking
                track_start = time.time()
                tracks = self.tracker.update(detections)
                self.metrics.tracking_times.append(time.time() - track_start)
                
                # Kinematics calculation
                kinematics_states = {}
                for track in tracks:
                    kin_state = self.kinematics.calculate(track, all_tracks=tracks)
                    kinematics_states[track.track_id] = kin_state
                
                # Collision detection
                col_start = time.time()
                collisions = self.collision_detector.detect(tracks, frame_idx, kinematics_states)
                # Filter duplicates is no longer needed with the state machine
                self.metrics.collision_times.append(time.time() - col_start)
                
                # Severity classification
                severity_assessments = {}
                for collision in collisions:
                    assessment = self.severity_classifier.classify(collision, kinematics_states, self.kinematics)
                    key = tuple(sorted(collision.track_ids))
                    severity_assessments[key] = assessment
                    
                    # Track accident
                    detected_accidents.append({
                        'frame': frame_idx,
                        'timestamp': format_timestamp(frame_idx, int(fps)),
                        'vehicle_ids': collision.track_ids,
                        'severity': assessment.level.value,
                        'score': assessment.score,
                        'confidence': collision.confidence,
                        'description': assessment.description
                    })
                    
                    self.metrics.accidents_detected += 1
                
                # Visualization
                if visualize:
                    output_frame = self.visualizer.draw_frame(
                        frame, tracks, collisions, severity_assessments,
                        frame_idx, total_frames
                    )
                else:
                    output_frame = frame
                
                # Write to output video
                if video_writer:
                    video_writer.write(output_frame)
                
                # Update metrics
                self.metrics.frame_count += 1
                frame_processing_time = time.time() - frame_start_time
                self.metrics.total_time += frame_processing_time
                
                # Update UI periodically
                # Frame callback more frequent for "video" feel
                if frame_idx % 5 == 0 and frame_callback:
                    try:
                        frame_callback(
                            frame=output_frame,
                            frame_idx=frame_idx,
                            accidents_in_frame=len(collisions)
                        )
                    except Exception as e:
                        self.logger.warning(f"Frame callback error: {e}")

                # Progress log and stats every 15 frames
                if frame_idx % 15 == 0:
                    fps_current = self.metrics.frame_count / self.metrics.total_time if self.metrics.total_time > 0 else 0
                    progress_percent = (frame_idx / total_frames) * 100 if total_frames > 0 else 0
                    
                    self.logger.info(
                        f"Progress: {frame_idx}/{total_frames} | "
                        f"FPS: {fps_current:.1f} | "
                        f"Accidents: {self.metrics.accidents_detected}"
                    )
                    
                    # Call progress callback
                    if progress_callback:
                        try:
                            progress_callback(
                                progress=progress_percent,
                                current_frame=frame_idx,
                                total_frames=total_frames,
                                accidents=self.metrics.accidents_detected,
                                fps=fps_current
                            )
                        except Exception as e:
                            self.logger.warning(f"Progress callback error: {e}")
                
                frame_idx += 1
        
        finally:
            cap.release()
            if video_writer:
                video_writer.release()
        
        self.logger.info("Video processing completed")
        
        # Generate results
        results = {
            'status': 'success',
            'input_video': video_path,
            'output_video': output_video_path,
            'total_frames': self.metrics.frame_count,
            'total_time_seconds': self.metrics.total_time,
            'fps': self.metrics.get_fps(),
            'vehicles_tracked': len(self.tracker.get_tracks(confirmed_only=False)),
            'accidents_detected': self.metrics.accidents_detected,
            'detected_accidents': detected_accidents,
            'metrics': self.metrics.get_report()
        }
        
        # Export results
        if export_json or export_csv:
            self._export_results(results, video_path, export_json, export_csv)
        
        return results
    
    def _export_results(
        self,
        results: Dict,
        video_path: str,
        export_json: bool = True,
        export_csv: bool = True
    ) -> None:
        """
        Export results to JSON and/or CSV.
        
        Args:
            results: Results dictionary
            video_path: Path to video file
            export_json: Whether to export JSON
            export_csv: Whether to export CSV
        """
        video_name = Path(video_path).stem
        output_dir = Path(self.config.get('export.output_dir', 'outputs/'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if export_json:
            json_path = output_dir / f"{video_name}_results.json"
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=2)
            self.logger.info(f"Results exported to JSON: {json_path}")
        
        if export_csv:
            csv_path = output_dir / f"{video_name}_events.csv"
            if results['detected_accidents']:
                with open(csv_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=results['detected_accidents'][0].keys())
                    writer.writeheader()
                    writer.writerows(results['detected_accidents'])
                self.logger.info(f"Events exported to CSV: {csv_path}")
    
    def process_frame(self, frame) -> Dict:
        """
        Process a single frame.
        
        Args:
            frame: Input frame (numpy array)
            
        Returns:
            Dictionary with frame results
        """
        # Detection
        detections = self.detector.detect(frame)
        
        # Tracking
        tracks = self.tracker.update(detections)
        
        # Kinematics
        kinematics_states = {}
        for track in tracks:
            kin_state = self.kinematics.calculate(track, all_tracks=tracks)
            kinematics_states[track.track_id] = kin_state
        
        # Collision detection
        collisions = self.collision_detector.detect(tracks, 0, kinematics_states)
        # Filter duplicates is no longer needed
        
        # Severity
        severity_assessments = {}
        for collision in collisions:
            assessment = self.severity_classifier.classify(collision, kinematics_states, self.kinematics)
            key = tuple(sorted(collision.track_ids))
            severity_assessments[key] = assessment
        
        return {
            'detections': detections,
            'tracks': tracks,
            'kinematics': kinematics_states,
            'collisions': collisions,
            'severity': severity_assessments
        }
