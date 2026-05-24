"""
Adaptive Learning System - Aprende y mejora automáticamente
"""

import json
from pathlib import Path
from typing import Dict, List
import numpy as np
from datetime import datetime

class AdaptiveLearner:
    """Sistema que aprende de cada análisis y auto-ajusta parámetros"""
    
    def __init__(self, history_file: str = "config/learning_history.json"):
        self.history_file = Path(history_file)
        self.history_file.parent.mkdir(exist_ok=True)
        self.history = self._load_history()
        self.current_params = self._calculate_optimal_params()
    
    def _load_history(self) -> List[Dict]:
        """Cargar histórico de análisis previos"""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def _save_history(self):
        """Guardar histórico actualizado"""
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def record_analysis(self, 
                        video_name: str,
                        accidents_detected: int,
                        total_frames: int,
                        confidence_threshold: float,
                        iou_threshold: float,
                        collision_sensitivity: float,
                        validation_frames: int = 15,
                        confirmed_score: float = 0.60,
                        false_positive_feedback: float = 0.5) -> None:
        """
        Registrar un análisis completado con parámetros extendidos.
        """
        record = {
            'timestamp': datetime.now().isoformat(),
            'video_name': video_name,
            'accidents_detected': accidents_detected,
            'total_frames': total_frames,
            'accident_rate': accidents_detected / total_frames if total_frames > 0 else 0,
            'confidence_threshold': confidence_threshold,
            'iou_threshold': iou_threshold,
            'collision_sensitivity': collision_sensitivity,
            'validation_frames': validation_frames,
            'confirmed_score': confirmed_score,
            'quality_score': false_positive_feedback  # 1=excelente, 0=pobre
        }
        self.history.append(record)
        self._save_history()
        self.current_params = self._calculate_optimal_params()
    
    def _calculate_optimal_params(self) -> Dict:
        """
        Calcular parámetros óptimos basado en histórico.
        Usa machine learning simple: weighted average de parámetros con mejor calidad.
        """
        if not self.history:
            return {
                'confidence_threshold': 0.35,
                'iou_threshold': 0.45,
                'collision_sensitivity': 0.6,
                'validation_frames': 12,
                'confirmed_score': 0.55,
                'learning_iterations': 0,
                'quality_history': []
            }
        
        # Convertir a array numpy para cálculos
        records = np.array([
            {
                'conf': r['confidence_threshold'],
                'iou': r['iou_threshold'],
                'coll': r['collision_sensitivity'],
                'quality': r['quality_score'],
                'accident_rate': r['accident_rate']
            }
            for r in self.history
        ], dtype=object)
        
        # Si tenemos poco historial, usar valores conservadores
        if len(records) < 3:
            return {
                'confidence_threshold': 0.30,
                'iou_threshold': 0.5,
                'collision_sensitivity': 0.5,
                'validation_frames': 10,
                'confirmed_score': 0.5,
                'learning_iterations': len(records),
                'quality_history': [r['quality_score'] for r in self.history]
            }
        
        # Extraer datos extendidos
        qualities = np.array([r['quality'] for r in records])
        confs = np.array([r['conf'] for r in records])
        ious = np.array([r['iou'] for r in records])
        colls = np.array([r['coll'] for r in records])
        v_frames = np.array([r.get('validation_frames', 15) for r in records])
        scores = np.array([r.get('confirmed_score', 0.60) for r in records])
        accident_rates = np.array([r['accident_rate'] for r in records])
        
        # Pesos basados en calidad y recencia
        weights = (qualities - qualities.min()) / (qualities.max() - qualities.min() + 0.0001) + 0.1
        recency_weights = np.linspace(0.5, 1.0, len(weights))
        weights = weights * recency_weights
        weights = weights / weights.sum()
        
        # Media ponderada
        optimal_conf = float(np.average(confs, weights=weights))
        optimal_iou = float(np.average(ious, weights=weights))
        optimal_coll = float(np.average(colls, weights=weights))
        optimal_v_frames = int(np.average(v_frames, weights=weights))
        optimal_score = float(np.average(scores, weights=weights))
        
        # Ajuste adaptativo según accidentes detectados
        mean_rate = np.mean(accident_rates)
        if mean_rate > 0.01: # Muchas detecciones -> endurecer
            optimal_v_frames = min(45, optimal_v_frames + 5)
            optimal_score = min(0.85, optimal_score + 0.05)
        elif mean_rate < 0.0001: # Pocas detecciones -> sensibilizar
            optimal_v_frames = max(5, optimal_v_frames - 3)
            optimal_score = max(0.40, optimal_score - 0.05)
            optimal_conf = max(0.20, optimal_conf - 0.05)
        
        return {
            'confidence_threshold': round(max(0.2, min(0.8, optimal_conf)), 2),
            'iou_threshold': round(max(0.2, min(0.9, optimal_iou)), 2),
            'collision_sensitivity': round(max(0.2, min(0.95, optimal_coll)), 2),
            'validation_frames': int(max(5, min(60, optimal_v_frames))),
            'confirmed_score': round(max(0.35, min(0.90, optimal_score)), 2),
            'learning_iterations': len(records),
            'quality_history': list(qualities),
            'last_quality': float(qualities[-1])
        }
    
    def get_current_params(self) -> Dict:
        """Obtener parámetros actuales optimizados"""
        return self.current_params
    
    def get_stats(self) -> Dict:
        """Obtener estadísticas del aprendizaje"""
        if not self.history:
            return {
                'total_analyses': 0,
                'avg_quality': 0,
                'best_quality': 0,
                'total_accidents_detected': 0,
                'learning_progress': 0
            }
        
        qualities = [r['quality_score'] for r in self.history]
        return {
            'total_analyses': len(self.history),
            'avg_quality': round(float(np.mean(qualities)), 2),
            'best_quality': round(float(np.max(qualities)), 2),
            'total_accidents_detected': sum(r['accidents_detected'] for r in self.history),
            'learning_iterations': len(self.history),
            'learning_progress': min(100, int((len(self.history) / 10) * 100))  # Mejora cada 10 análisis
        }
    
    def reset_learning(self):
        """Reiniciar todo el aprendizaje"""
        self.history = []
        self.history_file.unlink(missing_ok=True)
        self.current_params = self._calculate_optimal_params()
    
    def get_recommendations(self) -> str:
        """Obtener recomendaciones de mejora"""
        if len(self.history) < 3:
            return "🔄 Realizando primeros análisis para aprender parámetros óptimos..."
        
        stats = self.get_stats()
        quality = stats['avg_quality']
        
        if quality >= 0.8:
            return f"✅ Sistema optimizado (calidad: {quality:.0%}). Parámetros ajustados automáticamente."
        elif quality >= 0.6:
            return f"📈 Mejorando (calidad: {quality:.0%}). Continúa analizando para mejor precisión."
        else:
            return f"🔧 En optimización (calidad: {quality:.0%}). Procesando más videos para aprender."
