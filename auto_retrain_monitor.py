"""
Auto-Retraining System - Reentrenamiento automático basado en feedback
Monitorea calidad de análisis y reentrana el modelo cuando necesita mejoras
"""

import json
from pathlib import Path
import subprocess
import shutil
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class AutoRetrainingManager:
    """Gestiona re-entrenamiento automático basado en feedback del usuario"""
    
    def __init__(self):
        self.learning_history = Path("config/learning_history.json")
        self.retraining_log = Path("config/retraining_log.json")
        self.current_model = Path("yolov8n.pt")
        self.best_model = Path("models/best_model.pt")
        self.retraining_log.parent.mkdir(exist_ok=True)
    
    def check_need_retraining(self, threshold_quality=0.6, min_analyses=3):
        """
        Verificar si es necesario reentrenar basado en:
        - Calidad promedio de análisis
        - Tasa de falsos positivos
        """
        if not self.learning_history.exists():
            logger.info("📋 Sin historial previo, saltando chequeo de re-entrenamiento")
            return False
        
        try:
            with open(self.learning_history) as f:
                history = json.load(f)
        except:
            return False
        
        if len(history) < min_analyses:
            logger.info(f"📊 Análisis insuficientes ({len(history)}/{min_analyses})")
            return False
        
        # Calcular métricas recientes (últimos 5 análisis)
        recent = history[-5:]
        avg_quality = sum(r['quality_score'] for r in recent) / len(recent)
        false_positive_rate = sum(r['accident_rate'] for r in recent) / len(recent)
        
        logger.info(f"📈 Calidad promedio: {avg_quality:.2%}")
        logger.info(f"⚠️ Tasa de accidentes: {false_positive_rate:.2%}")
        
        # Criterios para re-entrenar
        needs_retraining = (
            avg_quality < threshold_quality or  # Calidad baja
            false_positive_rate > 0.05  # >5% de accidentes es alto
        )
        
        if needs_retraining:
            logger.warning(f"🚨 Calidad baja detectada: {avg_quality:.2%} - SE NECESITA RE-ENTRENAMIENTO")
        
        return needs_retraining
    
    def trigger_retraining(self, epochs=10, patience=3):
        """
        Disparar re-entrenamiento del modelo YOLO
        """
        logger.info("\n" + "="*60)
        logger.info("🤖 INICIANDO RE-ENTRENAMIENTO AUTOMÁTICO")
        logger.info("="*60)
        
        try:
            # Ejecutar script de entrenamiento rápido
            result = subprocess.run(
                ["python", "fast_training.py"],
                capture_output=True,
                text=True,
                timeout=1800  # 30 minutos máximo
            )
            
            logger.info(result.stdout)
            
            if result.returncode == 0:
                logger.info("✅ Re-entrenamiento completado exitosamente")
                
                # Copiar modelo mejorado
                best_trained = Path("models/best_model_trained.pt")
                if best_trained.exists():
                    shutil.copy2(best_trained, self.current_model)
                    logger.info(f"📦 Modelo actualizado: {self.current_model}")
                    
                    # Registrar
                    self._log_retraining_event('success', epochs)
                    return True
            else:
                logger.error(f"❌ Error en re-entrenamiento: {result.stderr}")
                self._log_retraining_event('failed', epochs, result.stderr)
                
        except subprocess.TimeoutExpired:
            logger.error("⏱️ Re-entrenamiento cancelado por timeout")
            self._log_retraining_event('timeout', epochs)
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            self._log_retraining_event('error', epochs, str(e))
        
        return False
    
    def _log_retraining_event(self, status, epochs, error_msg=""):
        """Registrar evento de re-entrenamiento"""
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'status': status,
            'epochs': epochs,
            'error': error_msg
        }
        
        log_history = []
        if self.retraining_log.exists():
            try:
                with open(self.retraining_log) as f:
                    log_history = json.load(f)
            except:
                pass
        
        log_history.append(log_data)
        
        with open(self.retraining_log, 'w') as f:
            json.dump(log_history, f, indent=2)
        
        logger.info(f"📝 Evento registrado: {status}")
    
    def get_retraining_stats(self):
        """Obtener estadísticas de re-entrenamiento"""
        if not self.retraining_log.exists():
            return {
                'total_retrainings': 0,
                'successful': 0,
                'failed': 0,
                'last_retraining': None
            }
        
        try:
            with open(self.retraining_log) as f:
                log_history = json.load(f)
        except:
            return {
                'total_retrainings': 0,
                'successful': 0,
                'failed': 0,
                'last_retraining': None
            }
        
        successful = sum(1 for r in log_history if r['status'] == 'success')
        failed = sum(1 for r in log_history if r['status'] != 'success')
        last = log_history[-1] if log_history else None
        
        return {
            'total_retrainings': len(log_history),
            'successful': successful,
            'failed': failed,
            'last_retraining': last['timestamp'] if last else None
        }


def monitor_and_retrain():
    """
    Script de monitoreo continuo
    Ejecutar en background para auto-reentrenamiento
    """
    manager = AutoRetrainingManager()
    
    logger.info("🔍 Iniciando monitoreo de calidad...")
    logger.info("⏰ Verificando cada análisis si se necesita re-entrenamiento\n")
    
    # Chequeo inicial
    if manager.check_need_retraining(threshold_quality=0.65):
        logger.info("🚀 Iniciando re-entrenamiento automático...\n")
        success = manager.trigger_retraining(epochs=20)
        
        if success:
            logger.info("✨ Modelo mejorado y disponible para próximos análisis\n")
        else:
            logger.warning("⚠️ Re-entrenamiento falló, continuando con modelo actual\n")
    else:
        logger.info("✅ Calidad de análisis es buena, no se necesita re-entrenamiento\n")
    
    # Mostrar estadísticas
    stats = manager.get_retraining_stats()
    if stats['total_retrainings'] > 0:
        logger.info("📊 Estadísticas de re-entrenamiento:")
        logger.info(f"   Total: {stats['total_retrainings']}")
        logger.info(f"   Exitosos: {stats['successful']}")
        logger.info(f"   Fallidos: {stats['failed']}")
        logger.info(f"   Último: {stats['last_retraining']}\n")


if __name__ == "__main__":
    monitor_and_retrain()
