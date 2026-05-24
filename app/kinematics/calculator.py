"""
Kinematics Module for calculating motion properties.
Computes velocity, acceleration, direction, and proximity metrics.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import math

from app.utils.config import Config, get_logger
from app.tracking.tracker import Track


@dataclass
class KinematicState:
    """Represents kinematic properties of a vehicle."""
    track_id: int
    
    # Position
    position: Tuple[float, float]  # (x, y) in pixels
    
    # Velocity (m/s)
    velocity_magnitude: float  # Speed
    velocity_vector: Tuple[float, float]  # (vx, vy)
    
    # Direction
    direction_angle: float  # degrees (0-360)
    
    # Acceleration (m/s²)
    acceleration_magnitude: float
    acceleration_vector: Tuple[float, float]  # (ax, ay)
    
    # Change rates
    velocity_change_rate: float  # m/s per frame
    direction_change_rate: float  # degrees per frame


class KinematicsCalculator:
    """
    Calculate kinematic properties of tracked vehicles.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the kinematics calculator.
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
        self.logger = get_logger()
        
        # Load configuration
        self.pixel_to_meter = self.config.get('kinematics.pixel_to_meter', 0.05)
        self.fps = self.config.get('kinematics.fps', 30)
        self.velocity_smoothing_window = self.config.get('kinematics.velocity_smoothing_window', 5)
        
        # Thresholds
        self.velocity_change_threshold = self.config.get('kinematics.velocity_change_threshold', 2.0)
        self.acceleration_threshold = self.config.get('kinematics.acceleration_threshold', 5.0)
        self.direction_change_threshold = self.config.get('kinematics.direction_change_threshold', 45.0)
        
        # Proximity
        self.min_proximity_distance = self.config.get('kinematics.min_proximity_distance', 3.0)
        self.critical_proximity_distance = self.config.get('kinematics.critical_proximity_distance', 1.5)
        
        # History for smoothing
        self.velocity_history: Dict[int, List[float]] = {}
        
        self.logger.info("KinematicsCalculator initialized")
    
    def calculate(self, track: Track, all_tracks: List[Track] = []) -> KinematicState:
        """
        Calculate kinematic properties for a track.
        
        Args:
            track: Track object
            all_tracks: Optional list of all current tracks for density analysis
            
        Returns:
            KinematicState object
        """
        position = track.current_center
        
        # Calculate velocity
        velocity_magnitude, velocity_vector = self._calculate_velocity(track)
        
        # Calculate acceleration
        acceleration_magnitude, acceleration_vector = self._calculate_acceleration(track)
        
        # Calculate direction
        direction_angle = self._calculate_direction(velocity_vector)
        
        # Calculate change rates
        velocity_change_rate = self._calculate_velocity_change_rate(track)
        direction_change_rate = self._calculate_direction_change_rate(track)
        
        kinematic_state = KinematicState(
            track_id=track.track_id,
            position=position,
            velocity_magnitude=velocity_magnitude,
            velocity_vector=velocity_vector,
            direction_angle=direction_angle,
            acceleration_magnitude=acceleration_magnitude,
            acceleration_vector=acceleration_vector,
            velocity_change_rate=velocity_change_rate,
            direction_change_rate=direction_change_rate
        )
        
        return kinematic_state

    def project_path(self, track: Track, num_frames: int = 30) -> List[Tuple[float, float]]:
        """
        Project the future path of a vehicle based on current velocity and acceleration.
        
        Args:
            track: Track object
            num_frames: How many frames to project into the future
            
        Returns:
            List of projected (x, y) coordinates
        """
        current_pos = track.current_center
        _, vel_vec = self._calculate_velocity(track)
        _, acc_vec = self._calculate_acceleration(track)
        
        # Convert pixels to meters for calculation, then back to pixels for projection
        vx, vy = vel_vec
        ax, ay = acc_vec
        
        projected_path = []
        for f in range(1, num_frames + 1):
            t = f / self.fps
            # s = ut + 0.5at^2
            dx_m = (vx * t) + (0.5 * ax * t**2)
            dy_m = (vy * t) + (0.5 * ay * t**2)
            
            # Convert back to pixels
            px = current_pos[0] + (dx_m / self.pixel_to_meter)
            py = current_pos[1] + (dy_m / self.pixel_to_meter)
            projected_path.append((px, py))
            
        return projected_path

    def estimate_scene_density(self, tracks: List[Track]) -> float:
        """
        Estimate traffic density in the current frame (0.0 to 1.0).
        High density usually means congestion.
        """
        if not tracks:
            return 0.0
            
        count = len([t for t in tracks if t.is_confirmed()])
        # Assume a max capacity of 20 vehicles for a typical urban CCTV view
        # This could be more sophisticated based on lane detection
        density = min(count / 20.0, 1.0)
        return density

    def calculate_energy_index(self, track: Track) -> float:
        """
        Estimate relative kinetic energy.
        Since we don't have mass, we use box area as a proxy for mass.
        E ~ Area * Velocity^2
        """
        vel_mag, _ = self._calculate_velocity(track)
        # Using bbox area as proxy for mass (simplified)
        bbox = track.to_tlbr()
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        
        return area * (vel_mag ** 2)

    def _calculate_velocity(self, track: Track) -> Tuple[float, Tuple[float, float]]:
        """
        Calculate velocity magnitude and vector.
        
        Args:
            track: Track object
            
        Returns:
            Tuple of (magnitude in m/s, (vx, vy) in m/s)
        """
        if len(track.position_history) < 2:
            return 0.0, (0.0, 0.0)
        
        # Get positions with smoothing window
        positions = list(track.position_history)
        window_size = min(self.velocity_smoothing_window, len(positions))
        
        if window_size < 2:
            return 0.0, (0.0, 0.0)
        
        # Calculate displacement over window
        start_pos = positions[-window_size]
        end_pos = positions[-1]
        
        dx_pixels = end_pos[0] - start_pos[0]
        dy_pixels = end_pos[1] - start_pos[1]
        
        # Convert to meters
        dx_meters = dx_pixels * self.pixel_to_meter
        dy_meters = dy_pixels * self.pixel_to_meter
        
        # Calculate time interval
        time_seconds = (window_size - 1) / self.fps
        
        if time_seconds == 0:
            return 0.0, (0.0, 0.0)
        
        # Calculate velocity
        vx = dx_meters / time_seconds
        vy = dy_meters / time_seconds
        magnitude = np.sqrt(vx**2 + vy**2)
        
        return magnitude, (vx, vy)
    
    def _calculate_acceleration(self, track: Track) -> Tuple[float, Tuple[float, float]]:
        """
        Calculate acceleration magnitude and vector.
        
        Args:
            track: Track object
            
        Returns:
            Tuple of (magnitude in m/s², (ax, ay) in m/s²)
        """
        # Get previous and current velocities
        history_key = f"vel_{track.track_id}"
        
        if history_key not in self.velocity_history:
            self.velocity_history[history_key] = []
        
        # Use smoothed velocity magnitude for acceleration
        mag, _ = self._calculate_velocity(track)
        self.velocity_history[history_key].append(mag)
        
        # Keep only recent history
        if len(self.velocity_history[history_key]) > 10:
            self.velocity_history[history_key] = self.velocity_history[history_key][-10:]
        
        # Need at least 2 velocity measurements
        if len(self.velocity_history[history_key]) < 2:
            return 0.0, (0.0, 0.0)
        
        # Calculate velocity change over last 2 frames
        vel_now = self.velocity_history[history_key][-1]
        vel_prev = self.velocity_history[history_key][-2]
        
        acceleration = (vel_now - vel_prev) * self.fps  # m/s²
        
        # For vector acceleration, we look at velocity vector differences
        # But this requires saving velocity vectors in history too.
        # Simplified: vector acceleration is in the direction of velocity
        _, (vx, vy) = self._calculate_velocity(track)
        v_mag = np.sqrt(vx**2 + vy**2)
        
        if v_mag < 1e-6:
            return abs(acceleration), (0.0, 0.0)
            
        ax = (vx / v_mag) * acceleration
        ay = (vy / v_mag) * acceleration
        
        return abs(acceleration), (ax, ay)
    
    def _calculate_direction(self, velocity_vector: Tuple[float, float]) -> float:
        """
        Calculate movement direction in degrees.
        
        Args:
            velocity_vector: (vx, vy) velocity vector
            
        Returns:
            Direction angle in degrees (0-360)
        """
        vx, vy = velocity_vector
        
        if abs(vx) < 1e-6 and abs(vy) < 1e-6:
            return 0.0
        
        # Calculate angle in radians
        angle_rad = math.atan2(vy, vx)
        
        # Convert to degrees
        angle_deg = math.degrees(angle_rad)
        
        # Normalize to 0-360
        if angle_deg < 0:
            angle_deg += 360
        
        return angle_deg
    
    def _calculate_velocity_change_rate(self, track: Track) -> float:
        """
        Calculate rate of velocity change.
        
        Args:
            track: Track object
            
        Returns:
            Velocity change rate (m/s per frame)
        """
        history_key = f"vel_{track.track_id}"
        
        if history_key not in self.velocity_history or len(self.velocity_history[history_key]) < 2:
            return 0.0
        
        vel_history = self.velocity_history[history_key]
        if len(vel_history) < 2:
            return 0.0
        
        # Rate of change of velocity
        rate = vel_history[-1] - vel_history[-2]
        return rate
    
    def _calculate_direction_change_rate(self, track: Track) -> float:
        """
        Calculate rate of direction change.
        
        Args:
            track: Track object
            
        Returns:
            Direction change rate (degrees per frame)
        """
        if len(track.position_history) < 3:
            return 0.0
        
        positions = list(track.position_history)
        
        # Get last three positions
        p1 = positions[-3]
        p2 = positions[-2]
        p3 = positions[-1]
        
        # Calculate angles
        angle1 = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
        angle2 = math.atan2(p3[1] - p2[1], p3[0] - p2[0])
        
        # Calculate difference
        angle_diff = math.degrees(angle2 - angle1)
        
        # Normalize to -180 to 180
        while angle_diff > 180:
            angle_diff -= 360
        while angle_diff < -180:
            angle_diff += 360
        
        return abs(angle_diff)
    
    def calculate_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """
        Calculate distance between two positions in meters.
        
        Args:
            pos1: (x1, y1) in pixels
            pos2: (x2, y2) in pixels
            
        Returns:
            Distance in meters
        """
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        
        distance_pixels = np.sqrt(dx**2 + dy**2)
        distance_meters = distance_pixels * self.pixel_to_meter
        
        return distance_meters
    
    def detect_sudden_deceleration(self, kinematic_state: KinematicState) -> bool:
        """
        Detect if vehicle is decelerating suddenly.
        
        Args:
            kinematic_state: Kinematic state object
            
        Returns:
            True if sudden deceleration detected
        """
        return kinematic_state.velocity_change_rate < -self.velocity_change_threshold
    
    def detect_sudden_acceleration(self, kinematic_state: KinematicState) -> bool:
        """
        Detect if vehicle is accelerating suddenly.
        
        Args:
            kinematic_state: Kinematic state object
            
        Returns:
            True if sudden acceleration detected
        """
        return kinematic_state.velocity_change_rate > self.velocity_change_threshold
    
    def detect_high_acceleration(self, kinematic_state: KinematicState) -> bool:
        """
        Detect if vehicle has high acceleration magnitude.
        
        Args:
            kinematic_state: Kinematic state object
            
        Returns:
            True if high acceleration detected
        """
        return kinematic_state.acceleration_magnitude > self.acceleration_threshold
    
    def detect_direction_change(self, kinematic_state: KinematicState) -> bool:
        """
        Detect if vehicle direction changes significantly.
        
        Args:
            kinematic_state: Kinematic state object
            
        Returns:
            True if significant direction change detected
        """
        return kinematic_state.direction_change_rate > self.direction_change_threshold
