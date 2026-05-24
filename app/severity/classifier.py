"""
Severity Classification Module.
Classifies detected collisions into severity levels.
"""

from typing import Optional, Dict, Tuple
from enum import Enum
from dataclasses import dataclass

from app.utils.config import Config, get_logger
from app.collision.detector import CollisionEvent
from app.kinematics.calculator import KinematicState, KinematicsCalculator


class SeverityLevel(Enum):
    """Enumeration of severity levels."""
    LEVE = "Leve"  # Light
    MODERADO = "Moderado"  # Moderate
    SEVERO = "Severo"  # Severe


@dataclass
class SeverityAssessment:
    """Assessment of accident severity."""
    level: SeverityLevel
    score: float  # 0-100
    confidence: float  # 0-1
    factors: Dict[str, float]  # Contributing factors with scores
    description: str  # Human-readable description


class SeverityClassifier:
    """
    Rule-based severity classification for traffic accidents.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize severity classifier.
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
        self.logger = get_logger()
        
        # Load thresholds
        self.severity_thresholds = self.config.get('severity.thresholds', {
            'leve_max': 30,
            'moderado_max': 65,
            'severo_min': 66
        })
        
        # Load weights
        self.weights = self.config.get('severity.weights', {
            'velocity_change': 0.25,
            'direction_change': 0.15,
            'proximity': 0.15,
            'deceleration': 0.20,
            'permanence_time': 0.15,
            'trajectory_distortion': 0.10
        })
        
        # Normalize weights to sum to 1.0
        total_weight = sum(self.weights.values())
        self.weights = {k: v / total_weight for k, v in self.weights.items()}
        
        # Load severity level thresholds
        self.severe_velocity_change = self.config.get('severity.severe_velocity_change', 4.0)
        self.moderate_velocity_change = self.config.get('severity.moderate_velocity_change', 2.0)
        
        self.severe_deceleration = self.config.get('severity.severe_deceleration', 7.0)
        self.moderate_deceleration = self.config.get('severity.moderate_deceleration', 3.5)
        
        self.logger.info("SeverityClassifier initialized")
    
    def classify(
        self,
        collision_event: CollisionEvent,
        kinematics_dict: Dict[int, KinematicState],
        kinematics_calc: KinematicsCalculator  # Passing calculator for energy index
    ) -> SeverityAssessment:
        """
        Classify severity of a collision event using impulse and energy metrics.
        """
        factors = {}
        
        # Get kinematic states for involved vehicles
        kin_states = [
            kinematics_dict.get(track_id) 
            for track_id in collision_event.track_ids
            if track_id in kinematics_dict
        ]
        
        if len(kin_states) < 2:
            return SeverityAssessment(
                level=SeverityLevel.LEVE,
                score=10.0,
                confidence=0.5,
                factors={},
                description="Datos incompletos para clasificar severidad"
            )
        
        # 1. Delta-V Impact Score (Normalized sudden velocity drop)
        dv_score = self._score_velocity_change(kin_states)
        factors['velocity_change'] = dv_score
        
        # 2. Relative Energy Index Dissipation
        # We check energy before vs after (approximated)
        energy_score = self._calculate_energy_impact_score(collision_event.track_ids, kinematics_calc)
        factors['energy_dissipation'] = energy_score
        
        # 3. Post-Impact Permanence (Are they still together or stationary?)
        permanence_score = self._score_permanence(collision_event)
        factors['post_impact_stasis'] = permanence_score
        
        # 4. Trajectory Distortion (Impulse indicator)
        distortion_score = self._score_direction_change(kin_states)
        factors['trajectory_distortion'] = distortion_score

        # Weighted combination
        weights = {
            'velocity_change': 0.40,
            'energy_dissipation': 0.30,
            'post_impact_stasis': 0.20,
            'trajectory_distortion': 0.10
        }
        
        final_score = sum(factors[f] * weights[f] for f in factors) * 100
        
        # Determine level
        if final_score <= 30:
            level = SeverityLevel.LEVE
        elif final_score <= 70:
            level = SeverityLevel.MODERADO
        else:
            level = SeverityLevel.SEVERO
            
        return SeverityAssessment(
            level=level,
            score=final_score,
            confidence=collision_event.confidence,
            factors=factors,
            description=self._generate_description(level, factors, final_score)
        )

    def _calculate_energy_impact_score(self, track_ids, calc) -> float:
        """Approximates energy dissipation during impact."""
        # This is a simplification. In a real system, we'd compare 
        # energy 5 frames before vs 5 frames after.
        return 0.5 # Placeholder for refined logic

    def _score_velocity_change(self, kin_states: list) -> float:
        # 5m/s drop is considered very high (approx 18km/h instant drop)
        max_change = max((abs(kin.velocity_change_rate) for kin in kin_states), default=0.0)
        return min(max_change / 5.0, 1.0)
        
    def _score_direction_change(self, kin_states: list) -> float:
        max_change = max((kin.direction_change_rate for kin in kin_states), default=0.0)
        return min(max_change / 90.0, 1.0) # 90 degree snap is max severity

    def _score_permanence(self, collision_event: CollisionEvent) -> float:
        return 1.0 if collision_event.vehicles_stopped_after else 0.2

    def _generate_description(self, level, factors, score) -> str:
        if level == SeverityLevel.LEVE:
            return f"Impacto leve detectado ({score:.1f}%). Movimiento lateral o frenada controlada."
        elif level == SeverityLevel.MODERADO:
            return f"Colisión moderada ({score:.1f}%). Pérdida significativa de energía y detención parcial."
        else:
            return f"ALERTA: Accidente SEVERO ({score:.1f}%). Impacto de alta energía con detención o deformación de trayectoria."
