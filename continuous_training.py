"""
Continuous Training System - Entrenamiento continuo hasta perfección
Combina datos sintéticos + reales de Kaggle para entrenar modelo YOLO
hasta alcanzar máxima precisión y mínimos falsos positivos
"""

import cv2
import numpy as np
from pathlib import Path
import torch
from ultralytics import YOLO
import yaml
import os
import shutil
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class ContinuousTrainer:
    """Entrena YOLO continuamente hasta alcanzar precisión objetivo"""
    
    def __init__(self, target_precision=0.95, max_epochs=100):
        self.target_precision = target_precision
        self.max_epochs = max_epochs
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dataset_dir = Path("yolo_training/dataset")
        self.model_path = Path("yolov8n.pt")
        logger.info(f"🚀 Entrenador continuo iniciado (Device: {self.device})")
        
    def prepare_dataset(self):
        """Preparar dataset YOLO con sintéticos + reales"""
        logger.info("📦 Preparando dataset combinado...")
        
        # Crear estructura YOLO
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        for split in ['train', 'val', 'test']:
            (self.dataset_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.dataset_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Crear dataset sintético mejorado (vehículos con variaciones)
        self._create_synthetic_dataset()
        
        # Procesar video real de Kaggle
        self._process_real_video()
        
        # Crear archivo yaml de configuración
        self._create_dataset_yaml()
        
        logger.info("✅ Dataset preparado")
    
    def _create_synthetic_dataset(self):
        """Generar dataset sintético con vehículos variados"""
        logger.info("🎨 Generando datos sintéticos mejorados...")
        
        num_images = 400  # Aumentado de 333
        images_per_split = {
            'train': int(num_images * 0.7),  # 70% train
            'val': int(num_images * 0.2),    # 20% val
            'test': int(num_images * 0.1)    # 10% test
        }
        
        idx = 0
        for split, count in images_per_split.items():
            for i in range(count):
                frame = self._generate_synthetic_frame(i)
                
                # Guardar imagen
                img_path = self.dataset_dir / split / 'images' / f"{split}_{idx:04d}.jpg"
                cv2.imwrite(str(img_path), frame)
                
                # Generar labels (YOLO format: class x_center y_center width height)
                label_path = self.dataset_dir / split / 'labels' / f"{split}_{idx:04d}.txt"
                self._save_yolo_labels(label_path, frame.shape[:2])
                
                idx += 1
        
        logger.info(f"✅ {num_images} imágenes sintéticas generadas")
    
    def _generate_synthetic_frame(self, seed):
        """Generar frame sintético con vehículos variados"""
        np.random.seed(seed)
        
        # Crear background (carretera)
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 100  # Gris
        cv2.rectangle(frame, (0, 200), (640, 480), (80, 80, 80), -1)  # Carretera oscura
        cv2.line(frame, (0, 340), (640, 340), (255, 255, 255), 2)  # Línea central
        
        # Generar 1-3 vehículos por imagen
        num_vehicles = np.random.randint(1, 4)
        for _ in range(num_vehicles):
            x = np.random.randint(50, 590)
            y = np.random.randint(220, 420)
            w = np.random.randint(40, 120)
            h = np.random.randint(30, 80)
            
            # Tipo de vehículo (color diferente)
            vehicle_type = np.random.choice(['car', 'truck', 'motorcycle'])
            if vehicle_type == 'truck':
                color = (0, 100, 200)  # Naranja
                w = max(w, 80)
            elif vehicle_type == 'motorcycle':
                color = (200, 0, 0)  # Azul
                w = max(w, 30)
            else:
                color = (0, 200, 200)  # Amarillo (car)
            
            # Dibujar vehículo
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, -1)
            
            # Añadir detalles (windows)
            cv2.rectangle(frame, (x+5, y+5), (x+w-5, y+h-15), (100, 100, 100), -1)
        
        # Añadir ruido y variaciones
        noise = np.random.normal(0, 10, frame.shape).astype(np.uint8)
        frame = cv2.add(frame, noise)
        
        return frame
    
    def _save_yolo_labels(self, label_path, img_shape):
        """Guardar labels en formato YOLO"""
        with open(label_path, 'w') as f:
            # Crear 1-3 anotaciones por imagen
            num_objs = np.random.randint(1, 4)
            for _ in range(num_objs):
                class_id = np.random.randint(0, 4)  # 0=car, 1=motorcycle, 2=bus, 3=truck
                x_center = np.random.uniform(0.1, 0.9)
                y_center = np.random.uniform(0.3, 0.9)
                width = np.random.uniform(0.1, 0.3)
                height = np.random.uniform(0.1, 0.3)
                f.write(f"{class_id} {x_center:.3f} {y_center:.3f} {width:.3f} {height:.3f}\n")
    
    def _process_real_video(self):
        """Extraer frames del video real de Kaggle"""
        logger.info("🎬 Procesando video real de Kaggle...")
        
        video_path = Path("datasets/videoplayback (online-video-cutter.com).mp4")
        if not video_path.exists():
            logger.warning("⚠️ Video Kaggle no encontrado, saltando procesamiento de datos reales")
            return
        
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Extraer frames cada N frames para balance dataset
        frame_step = max(1, total_frames // 150)  # ~150 frames del video real
        frame_count = 0
        extracted = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_step == 0:
                # Redimensionar
                frame = cv2.resize(frame, (640, 480))
                
                # Guardar en train set
                split = 'train' if extracted % 5 != 0 else 'val'  # 80/20 split
                img_path = self.dataset_dir / split / 'images' / f"real_{extracted:04d}.jpg"
                cv2.imwrite(str(img_path), frame)
                
                # Crear labels simplificados para datos reales (detección genérica)
                label_path = self.dataset_dir / split / 'labels' / f"real_{extracted:04d}.txt"
                with open(label_path, 'w') as f:
                    # Asumir vehículos en área central
                    f.write(f"0 0.5 0.5 0.3 0.3\n")  # Car genérico en centro
                
                extracted += 1
            
            frame_count += 1
        
        cap.release()
        logger.info(f"✅ {extracted} frames reales extraídos")
    
    def _create_dataset_yaml(self):
        """Crear archivo de configuración del dataset"""
        dataset_config = {
            'path': str(self.dataset_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': 4,
            'names': ['car', 'motorcycle', 'bus', 'truck']
        }
        
        yaml_path = self.dataset_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        logger.info(f"✅ Configuración dataset guardada en {yaml_path}")
    
    def train_continuous(self):
        """Entrenar continuamente hasta alcanzar precisión objetivo"""
        logger.info("🎓 Iniciando entrenamiento continuo...")
        
        # Preparar dataset
        self.prepare_dataset()
        
        # Cargar modelo
        model = YOLO(self.model_path)
        
        best_precision = 0.0
        best_model_path = Path("models/best_model.pt")
        best_model_path.parent.mkdir(exist_ok=True)
        
        # Loop de entrenamiento
        for epoch in range(1, self.max_epochs + 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"🔄 ÉPOCA {epoch}/{self.max_epochs}")
            logger.info(f"{'='*60}")
            
            # Entrenar
            results = model.train(
                data=str(self.dataset_dir / 'data.yaml'),
                epochs=1,  # Una época por iteración
                imgsz=640,
                batch=8 if self.device == 'cpu' else 16,
                patience=5,
                device=self.device,
                save=True,
                verbose=True,
                workers=0,
                augment=True,  # Data augmentation
                mosaic=1.0,  # Mosaic augmentation
                flipud=0.5,  # Flip vertical
                fliplr=0.5,  # Flip horizontal
                degrees=10,  # Rotación
                translate=0.1,  # Traslación
                scale=0.5,  # Escala
            )
            
            # Extraer métricas
            if hasattr(results, 'results_dict'):
                metrics = results.results_dict
                precision = metrics.get('metrics/precision(B)', 0)
                recall = metrics.get('metrics/recall(B)', 0)
                map50 = metrics.get('metrics/mAP50(B)', 0)
            else:
                precision = recall = map50 = 0
            
            logger.info(f"📊 Precisión: {precision:.2%} | Recall: {recall:.2%} | mAP50: {map50:.2%}")
            
            # Guardar si es mejor
            if precision > best_precision:
                best_precision = precision
                shutil.copy2(model.trainer.best.replace('\\', '/'), best_model_path)
                logger.info(f"✨ ¡MEJOR MODELO! Precisión: {best_precision:.2%} guardado")
            
            # Verificar si alcanzó objetivo
            if best_precision >= self.target_precision:
                logger.info(f"\n🎉 ¡ENTRENAMIENTO COMPLETO!")
                logger.info(f"✅ Precisión objetivo alcanzada: {best_precision:.2%}")
                logger.info(f"📁 Modelo guardado en: {best_model_path}")
                break
            
            # Mostrar progreso
            remaining = self.target_precision - best_precision
            logger.info(f"📈 Progreso: {best_precision:.2%} (Falta {remaining:.2%} para objetivo)")
        
        logger.info(f"\n✅ Entrenamiento finalizado")
        logger.info(f"   Mejor precisión alcanzada: {best_precision:.2%}")
        logger.info(f"   Modelo guardado: {best_model_path}")
        
        return best_model_path
    
    def validate_model(self, model_path):
        """Validar modelo en dataset de test"""
        logger.info("\n🔍 Validando modelo...")
        
        model = YOLO(model_path)
        results = model.val(data=str(self.dataset_dir / 'data.yaml'))
        
        logger.info("✅ Validación completada")
        return results


def main():
    """Ejecutar entrenamiento continuo"""
    trainer = ContinuousTrainer(
        target_precision=0.90,  # 90% de precisión como objetivo
        max_epochs=50
    )
    
    # Entrenar
    best_model = trainer.train_continuous()
    
    # Validar
    trainer.validate_model(best_model)


if __name__ == "__main__":
    main()
