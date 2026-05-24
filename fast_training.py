"""
Fast Training - Entrenamiento rápido usando datos existentes
Optimizado para mejorar modelo con mínimas épocas pero máxima efectividad
"""

import torch
from pathlib import Path
from ultralytics import YOLO
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def fast_training():
    """Entrenamiento rápido con parámetros optimizados"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"📱 Device: {device}")
    
    # Cargar modelo entrenado
    model_path = Path("yolov8n.pt")
    if not model_path.exists():
        logger.error(f"❌ Modelo no encontrado: {model_path}")
        return False
    
    logger.info(f"📦 Cargando modelo: {model_path}")
    model = YOLO(str(model_path))
    
    # Dataset
    dataset_yaml = Path("yolo_training/dataset/data.yaml")
    
    logger.info("\n" + "="*60)
    logger.info("🚀 INICIANDO ENTRENAMIENTO RÁPIDO")
    logger.info("="*60)
    
    try:
        # Entrenar con parámetros optimizados para CPU
        results = model.train(
            data=str(dataset_yaml),
            epochs=15,  # 15 épocas (quick but effective)
            imgsz=640,
            batch=4 if device == 'cpu' else 16,  # Batch pequeño para CPU
            patience=3,  # Early stopping
            device=device,
            save=True,
            verbose=True,
            workers=0,
            
            # Augmentation params para reducir falsos positivos
            augment=True,
            mosaic=1.0,
            mixup=0.1,  # Mixup augmentation
            flipud=0.5,
            fliplr=0.5,
            degrees=15,
            translate=0.15,
            scale=0.5,
            hsv_h=0.015,  # Color augmentation
            hsv_s=0.7,
            hsv_v=0.4,
            
            # Learning rate
            lr0=0.01,  # Learning rate inicial
            lrf=0.01,  # Learning rate final
            momentum=0.937,
            
            # Loss weights
            cls=0.5,  # Classification loss weight
            conf=1.0,  # Confidence loss weight
            iou=0.7,   # IoU loss weight
        )
        
        logger.info("\n" + "="*60)
        logger.info("✅ ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
        logger.info("="*60)
        
        # Obtener best model
        best_model_path = Path(model.trainer.best_path)
        logger.info(f"📦 Mejor modelo: {best_model_path}")
        
        # Copiar a ubicación estándar
        import shutil
        best_save_path = Path("models/best_model_trained.pt")
        best_save_path.parent.mkdir(exist_ok=True)
        shutil.copy2(best_model_path, best_save_path)
        logger.info(f"✨ Modelo guardado: {best_save_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error en entrenamiento: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = fast_training()
    exit(0 if success else 1)
