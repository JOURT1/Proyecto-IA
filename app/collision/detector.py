"""
Collision Detection Module.
Implements rule-based collision detection logic.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

from app.utils.config import Config, get_logger
from app.tracking.tracker import Track
from app.kinematics.calculator import KinematicsCalculator, KinematicState


from enum import Enum
import time

class DetectionState(Enum):
    """States of a collision event."""
    NONE = 0
    POTENTIAL = 1
    VALIDATING = 2
    CONFIRMED = 3
    POST_IMPACT = 4

@dataclass
class CollisionPairState:
    """Tracks the evolution of a potential collision between a pair of vehicles."""
    track_ids: Tuple[int, int]
    state: DetectionState = DetectionState.NONE
    frames_in_state: int = 0
    scores_history: List[float] = None
    start_frame: int = 0
    confirmed: bool = False
    cooldown_until: float = 0.0
    
    def __post_init__(self):
        if self.scores_history is None:
            self.scores_history = []

@dataclass
class CollisionEvent:
    """Represents a detected collision event."""
    frame_number: int
    track_ids: List[int]
    confidence: float
    signals: List[str]
    distance_at_impact: float
    relative_velocity: float
    vehicles_stopped_after: bool
    severity_score: float = 0.0
    
    def __hash__(self):
        return hash((self.frame_number, tuple(sorted(self.track_ids))))
    
    def __eq__(self, other):
        return (self.frame_number == other.frame_number and 
                set(self.track_ids) == set(other.track_ids))


class CollisionDetector:
    """
    Advanced state-based collision detection with temporal validation 
    and kinematic scoring.
    """
    
    def __init__(self, config: Optional[Config] = None, kinematics: Optional[KinematicsCalculator] = None):
        self.config = config or Config()
        self.logger = get_logger()
        self.kinematics = kinematics or KinematicsCalculator(config)
        
        # Scoring Weights
        self.weights = {
            'proximity': self.config.get('collision.weights.proximity', 0.40),
            'convergence': self.config.get('collision.weights.convergence', 0.35),
            'velocity_drop': self.config.get('collision.weights.velocity_drop', 0.15),
            'direction_change': self.config.get('collision.weights.direction_change', 0.10)
        }
        
        # Thresholds from Config - Highly reactive to ensure real crashes are captured
        self.score_threshold_potential = self.config.get('collision.potential_threshold', 0.35)
        self.score_threshold_confirmed = self.config.get('collision.confirmed_threshold', 0.55) # Relaxed for recall
        self.validation_frames_required = self.config.get('collision.collision_confirmation_frames', 8)  # Faster reaction
        self.alert_cooldown_seconds = self.config.get('collision.alert_cooldown', 600)
        self.min_movement_speed = self.config.get('collision.ignore_slow_collisions_below_velocity', 0.5)
        
        # State tracking
        self.pair_states: Dict[Tuple[int, int], CollisionPairState] = {}
        self.active_events: List[CollisionEvent] = []
        self.location_cooldowns: List[Dict] = [] # List of {pos: (x,y), time: float}
        
        self.logger.info("Advanced CollisionDetector initialized")
        
    def apply_optimal_params(self, params: Dict):
        """Apply parameters learned by the AdaptiveLearner."""
        self.score_threshold_potential = params.get('confidence_threshold', self.score_threshold_potential)
        self.score_threshold_confirmed = params.get('confirmed_score', self.score_threshold_confirmed)
        self.validation_frames_required = params.get('validation_frames', self.validation_frames_required)
        
        # Adjust weights based on sensitivity
        sens = params.get('collision_sensitivity', 0.5)
        self.weights['proximity'] = 0.3 + (0.2 * sens)
        self.weights['convergence'] = 0.3 + (0.2 * sens)
        
        self.logger.info(f"Self-Adjusted: Conf={self.score_threshold_confirmed}, Frames={self.validation_frames_required}")
    
    def detect(
        self,
        tracks: List[Track],
        frame_number: int,
        kinematics_states: Dict[int, KinematicState]
    ) -> List[CollisionEvent]:
        """
        Detect collisions using multi-frame validation.
        """
        current_frame_events = []
        
        # Cleanup old pairs not present in current tracks
        active_ids = {t.track_id for t in tracks}
        self.pair_states = {k: v for k, v in self.pair_states.items() 
                           if k[0] in active_ids and k[1] in active_ids}
        
        # Scene context
        density = self.kinematics.estimate_scene_density(tracks)
        
        # Analyze pairwise interactions
        for i in range(len(tracks)):
            for j in range(i + 1, len(tracks)):
                track_a, track_b = tracks[i], tracks[j]
                
                if not track_a.is_confirmed() or not track_b.is_confirmed():
                    continue
                
                pair_key = tuple(sorted((track_a.track_id, track_b.track_id)))
                if pair_key not in self.pair_states:
                    self.pair_states[pair_key] = CollisionPairState(track_ids=pair_key, start_frame=frame_number)
                
                state_obj = self.pair_states[pair_key]
                
                # Cooldown check
                if time.time() < state_obj.cooldown_until:
                    continue
                
                # 1. Calculate Comprehensive Score
                kin_a = kinematics_states.get(track_a.track_id)
                kin_b = kinematics_states.get(track_b.track_id)
                if not kin_a or not kin_b: continue
                
                score, active_signals = self._calculate_collision_score(track_a, track_b, kin_a, kin_b, density)
                state_obj.scores_history.append(score)
                if len(state_obj.scores_history) > 30: state_obj.scores_history.pop(0)
                
                # AUTO-ADJUST: High-Risk Hunting Mode
                # If path convergence is high, temporarily lower the confirmed threshold for THIS pair
                pair_threshold = self.score_threshold_confirmed
                if 'path_convergence' in active_signals:
                    pair_threshold *= 0.8  # 20% more sensitive in high-risk zones
                
                # 2. State Machine Logic
                new_event = self._update_state_machine(state_obj, score, active_signals, kin_a, kin_b, frame_number, pair_threshold)
                if new_event:
                    # Global Location Cooldown check
                    avg_pos = ((kin_a.position[0] + kin_b.position[0])/2, (kin_a.position[1] + kin_b.position[1])/2)
                    is_duplicate_loc = False
                    for cd in self.location_cooldowns:
                        if self.kinematics.calculate_distance(avg_pos, cd['pos']) < 10.0: # 10 meters
                            is_duplicate_loc = True
                            break
                    
                    if not is_duplicate_loc:
                        current_frame_events.append(new_event)
                        state_obj.cooldown_until = time.time() + self.alert_cooldown_seconds
                        self.location_cooldowns.append({'pos': avg_pos, 'time': time.time() + self.alert_cooldown_seconds})
                
        # Cleanup location cooldowns
        self.location_cooldowns = [cd for cd in self.location_cooldowns if time.time() < cd['time']]
        
        return current_frame_events

    def _calculate_collision_score(
        self, 
        t_a: Track, t_b: Track, 
        k_a: KinematicState, k_b: KinematicState,
        density: float
    ) -> Tuple[float, List[str]]:
        """Calculate a weighted score based on multiple kinematic features."""
        signals = []
        score = 0.0
        
        # A. Proximity Score (Inverse exponential)
        dist = self.kinematics.calculate_distance(k_a.position, k_b.position)
        # Critical if < 1.5m, Ignored if > 8.0m (extended range)
        prox_score = max(0, min(1.0, 1.0 - (dist - 0.5) / 7.5))
        if dist < 1.5:
            prox_score = 1.0 # Force max proximity score if extremely close
            signals.append('near_contact_zone')
            score += 0.2 # Direct bonus for risk zone
        
        if prox_score > 0.8: signals.append('extreme_proximity')
        score += prox_score * self.weights['proximity']
        
        # B. Path Convergence Score
        convergence = self._analyze_convergence(t_a, t_b, k_a, k_b)
        if convergence > 0.7: signals.append('path_convergence')
        score += convergence * self.weights['convergence']
        
        # C. Physical Overlap (IoU) - CRITICAL FOR PRECISION
        iou = self._calculate_iou(t_a.current_bbox, t_b.current_bbox)
        if iou > 0:
            score += 0.3  # Bonus for actual overlap
            signals.append('physical_contact')
        
        # D. Delta-V (Velocity Drop) Score
        dv_a = max(0, -k_a.velocity_change_rate) / 2.5
        dv_b = max(0, -k_b.velocity_change_rate) / 2.5
        vel_drop_score = min(1.0, (dv_a + dv_b))
        if vel_drop_score > 0.6: signals.append('impact_deceleration')
        score += vel_drop_score * self.weights['velocity_drop']
        
        # E. Direction Change (Delta-Theta)
        dt_a = k_a.direction_change_rate / 60.0
        dt_b = k_b.direction_change_rate / 60.0
        dir_score = min(1.0, (dt_a + dt_b))
        if dir_score > 0.6: signals.append('impact_direction_shift')
        score += dir_score * self.weights['direction_change']
        
        # F. High-Precision Sanitization
        # If no overlap AND distance > 2m, cap score at 0.75 (can't reach confirmed)
        if iou == 0 and dist > 2.0:
            score = min(0.75, score)
            
        # G. Congestion Penalty
        avg_speed = (k_a.velocity_magnitude + k_b.velocity_magnitude) / 2.0
        if avg_speed < self.min_movement_speed:
            return 0.0, []
            
        if density > 0.6 and avg_speed < 2.5:
            score *= 0.4  # Even more strict in traffic
            signals.append('congestion_filtered')
            
        return score, signals

    def _analyze_convergence(self, t_a: Track, t_b: Track, k_a: KinematicState, k_b: KinematicState) -> float:
        """Analyze if projections of paths will intersect within a time window."""
        path_a = self.kinematics.project_path(t_a, num_frames=30)
        path_b = self.kinematics.project_path(t_b, num_frames=30)
        
        min_dist_meters = float('inf')
        for i in range(min(len(path_a), len(path_b))):
            d = self.kinematics.calculate_distance(path_a[i], path_b[i])
            if d < min_dist_meters:
                min_dist_meters = d
        
        # If they get closer than 0.5m in projected path, high convergence
        convergence = max(0, min(1.0, 1.0 - (min_dist_meters / 3.0)))
        
        # Discard if moving in same direction (parallel)
        dot_product = np.dot(k_a.velocity_vector, k_b.velocity_vector)
        mags = k_a.velocity_magnitude * k_b.velocity_magnitude
        if mags > 0.1:
            cos_theta = dot_product / mags
            if cos_theta > 0.9: # Moving in same direction within ~25 degrees
                convergence *= 0.2
                
        return convergence

    def _update_state_machine(
        self, state_obj: CollisionPairState, 
        current_score: float, signals: List[str],
        kin_a: KinematicState, kin_b: KinematicState,
        frame: int,
        pair_threshold: Optional[float] = None
    ) -> Optional[CollisionEvent]:
        """Update detection state and confirm events if persistence is met."""
        
        target_confirm_threshold = pair_threshold or self.score_threshold_confirmed
        
        if state_obj.state == DetectionState.NONE:
            if current_score > self.score_threshold_potential:
                state_obj.state = DetectionState.POTENTIAL
                state_obj.frames_in_state = 1
        
        elif state_obj.state == DetectionState.POTENTIAL:
            if current_score > self.score_threshold_potential:
                state_obj.frames_in_state += 1
                if state_obj.frames_in_state >= 3:
                    state_obj.state = DetectionState.VALIDATING
            else:
                state_obj.state = DetectionState.NONE
        
        elif state_obj.state == DetectionState.VALIDATING:
            if current_score > self.score_threshold_potential:
                state_obj.frames_in_state += 1
                # Check for confirmation
                avg_recent_score = sum(state_obj.scores_history[-15:]) / min(15, len(state_obj.scores_history))
                if state_obj.frames_in_state >= self.validation_frames_required and avg_recent_score > target_confirm_threshold:
                    state_obj.state = DetectionState.CONFIRMED
                    return self._create_event(state_obj, avg_recent_score, signals, kin_a, kin_b, frame)
            else:
                # One frame drop is fine, but reset if it persists
                state_obj.frames_in_state = max(0, state_obj.frames_in_state - 2)
                if state_obj.frames_in_state == 0:
                    state_obj.state = DetectionState.NONE
        
        elif state_obj.state == DetectionState.CONFIRMED:
            # Once confirmed, stay in this state to avoid re-triggering until vehicles move apart
            dist = self.kinematics.calculate_distance(kin_a.position, kin_b.position)
            if dist > 10.0: # Vehicles cleared the scene
                state_obj.state = DetectionState.NONE
                    
        return None

    def _calculate_iou(self, box1: Tuple, box2: Tuple) -> float:
        """Compute IoU between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        xi1, yi1 = max(x1_1, x1_2), max(y1_1, y1_2)
        xi2, yi2 = min(x2_1, x2_2), min(y2_1, y2_2)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    def _create_event(self, state, score, signals, k_a, k_b, frame) -> CollisionEvent:
        dist = self.kinematics.calculate_distance(k_a.position, k_b.position)
        rel_vel = np.sqrt((k_a.velocity_vector[0] - k_b.velocity_vector[0])**2 + 
                         (k_a.velocity_vector[1] - k_b.velocity_vector[1])**2)
        
        return CollisionEvent(
            frame_number=frame,
            track_ids=list(state.track_ids),
            confidence=score,
            signals=signals,
            distance_at_impact=dist,
            relative_velocity=rel_vel,
            vehicles_stopped_after=(k_a.velocity_magnitude < 0.2 and k_b.velocity_magnitude < 0.2)
        )
