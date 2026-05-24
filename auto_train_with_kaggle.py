"""
Sistema Automático de Descarga y Entrenamiento de Kaggle
Descarga datos de accidentes reales y entrena el modelo YOLO v8
"""

import os
import sys
import json
from pathlib import Path
import subprocess
import cv2
import shutil
from datetime import datetime

class KaggleAutoTrainer:
    def __init__(self):
        self.project_root = Path.cwd()
        self.kaggle_config = Path.home() / ".kaggle" / "kaggle.json"
        self.datasets_dir = self.project_root / "datasets"
        self.training_data_dir = self.project_root / "datasets" / "training_data"
        self.models_dir = self.project_root / "models"
        
    def check_kaggle_credentials(self):
        """Verificar si existen credenciales de Kaggle"""
        if self.kaggle_config.exists():
            print("✅ Credenciales de Kaggle encontradas")
            return True
        else:
            print("⚠️  No se encontraron credenciales de Kaggle")
            print("📖 Para descargar datos reales de Kaggle:")
            print("   1. Ve a: https://www.kaggle.com/account/api")
            print("   2. Haz clic en 'Create New API Token'")
            print("   3. Se descargará kaggle.json")
            print(f"   4. Guárdalo en: {self.kaggle_config.parent}")
            print()
            return False
    
    def download_kaggle_dataset(self):
        """Descargar dataset de accidentes de Kaggle"""
        if not self.check_kaggle_credentials():
            print("⏭️  Saltando descarga de Kaggle...")
            return False
        
        try:
            print("\n🔄 Descargando dataset de Kaggle...")
            print("   Dataset: accident-detection-in-videos")
            
            os.makedirs(self.datasets_dir, exist_ok=True)
            
            # Descargar dataset
            cmd = [
                "kaggle", "datasets", "download",
                "-d", "usmanabbasi/accident-detection-in-videos",
                "-p", str(self.datasets_dir)
            ]
            
            subprocess.run(cmd, check=True)
            print("✅ Dataset descargado correctamente")
            
            # Extraer ZIP
            zip_file = self.datasets_dir / "accident-detection-in-videos.zip"
            if zip_file.exists():
                print("📦 Extrayendo archivos...")
                shutil.unpack_archive(zip_file, self.datasets_dir)
                zip_file.unlink()
                print("✅ Archivos extraídos")
                return True
        
        except subprocess.CalledProcessError as e:
            print(f"❌ Error descargando: {e}")
            return False
    
    def extract_frames_from_videos(self, max_frames_per_video=100):
        """Extraer frames de los videos para crear dataset de entrenamiento"""
        print("\n📹 Extrayendo frames de videos...")
        
        os.makedirs(self.training_data_dir, exist_ok=True)
        
        # Buscar todos los videos MP4
        video_files = list(self.datasets_dir.glob("**/*.mp4"))
        
        if not video_files:
            print("⚠️  No se encontraron videos MP4")
            return False
        
        print(f"📊 Se encontraron {len(video_files)} video(s)")
        
        frame_count = 0
        for video_idx, video_path in enumerate(video_files, 1):
            print(f"\n  [{video_idx}/{len(video_files)}] Procesando: {video_path.name}")
            
            try:
                cap = cv2.VideoCapture(str(video_path))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                print(f"      Frames: {total_frames} | FPS: {fps:.1f}")
                
                # Calcular intervalo de muestreo
                interval = max(1, total_frames // max_frames_per_video)
                extracted = 0
                
                frame_id = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Guardar cada N frames
                    if frame_id % interval == 0 and extracted < max_frames_per_video:
                        frame_path = self.training_data_dir / f"frame_{video_idx}_{extracted:04d}.jpg"
                        cv2.imwrite(str(frame_path), frame)
                        extracted += 1
                        frame_count += 1
                    
                    frame_id += 1
                
                cap.release()
                print(f"      ✓ Extraídos {extracted} frames")
            
            except Exception as e:
                print(f"      ❌ Error: {e}")
                continue
        
        print(f"\n✅ Total de frames extraídos: {frame_count}")
        return frame_count > 0
    
    def create_training_dataset_yaml(self):
        """Crear archivo YAML para entrenamiento de YOLO"""
        print("\n⚙️  Creando configuración de entrenamiento...")
        
        dataset_yaml = """
path: datasets/training_data  # dataset root
train: images/train  # train images (relative to 'path')
val: images/val      # val images (relative to 'path')

nc: 4  # number of classes
names: ['car', 'motorcycle', 'bus', 'truck']
"""
        
        config_path = self.project_root / "config" / "training_dataset.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            f.write(dataset_yaml)
        
        print(f"✅ Configuración guardada: {config_path}")
        return config_path
    
    def train_yolo_model(self, epochs=10, batch_size=16):
        """Entrenar modelo YOLO v8 con datos reales"""
        print("\n🤖 Iniciando entrenamiento YOLO v8...")
        print(f"   Épocas: {epochs}")
        print(f"   Batch size: {batch_size}")
        print()
        
        try:
            # Importar YOLO
            from ultralytics import YOLO
            
            # Cargar modelo pre-entrenado
            print("📥 Cargando modelo YOLO v8 nano...")
            model = YOLO('yolov8n.pt')
            
            # Entrenar
            print("🔧 Entrenando modelo...")
            results = model.train(
                data='config/settings.yaml',  # Usar nuestro dataset
                epochs=epochs,
                imgsz=640,
                batch=batch_size,
                patience=5,
                save=True,
                device=0,  # GPU si está disponible
                verbose=True,
                project='runs/detect',
                name='traffic_accident_v1'
            )
            
            print("\n✅ Entrenamiento completado")
            print(f"📁 Resultados guardados en: runs/detect/traffic_accident_v1")
            
            # Copiar mejor modelo
            best_model = Path("runs/detect/traffic_accident_v1/weights/best.pt")
            if best_model.exists():
                shutil.copy(best_model, self.models_dir / "best_trained_model.pt")
                print(f"💾 Mejor modelo guardado: {self.models_dir / 'best_trained_model.pt'}")
            
            return True
        
        except Exception as e:
            print(f"❌ Error en entrenamiento: {e}")
            return False
    
    def run_full_pipeline(self):
        """Ejecutar pipeline completo: descargar -> entrenar"""
        print("=" * 70)
        print("🚀 PIPELINE AUTOMÁTICO: DESCARGA Y ENTRENAMIENTO")
        print("=" * 70)
        print()
        
        # Paso 1: Intentar descargar de Kaggle
        print("PASO 1: Descargando datos de Kaggle")
        print("-" * 70)
        kaggle_success = self.download_kaggle_dataset()
        
        # Paso 2: Extraer frames
        print("\n\nPASO 2: Preparando dataset de entrenamiento")
        print("-" * 70)
        frames_extracted = self.extract_frames_from_videos()
        
        if not frames_extracted:
            print("⚠️  No hay frames para entrenar")
            return False
        
        # Paso 3: Crear configuración
        print("\n\nPASO 3: Creando configuración")
        print("-" * 70)
        self.create_training_dataset_yaml()
        
        # Paso 4: Entrenar modelo
        print("\n\nPASO 4: Entrenando modelo YOLO")
        print("-" * 70)
        training_success = self.train_yolo_model(epochs=5, batch_size=8)
        
        # Resumen final
        print("\n\n" + "=" * 70)
        print("✅ PIPELINE COMPLETADO")
        print("=" * 70)
        print(f"Kaggle descargado: {'✅ Sí' if kaggle_success else '⚠️  No (usar local)'}")
        print(f"Frames extraídos: {'✅ Sí' if frames_extracted else '❌ No'}")
        print(f"Entrenamiento: {'✅ Exitoso' if training_success else '❌ Fallido'}")
        print()
        
        return True


def main():
    """Función principal"""
    trainer = KaggleAutoTrainer()
    
    # Verificar si Kaggle está instalado
    try:
        import kaggle
        print("✅ Kaggle CLI instalado")
    except ImportError:
        print("⚠️  Instalando Kaggle CLI...")
        os.system("pip install kaggle")
    
    # Ejecutar pipeline
    trainer.run_full_pipeline()


if __name__ == "__main__":
    main()
