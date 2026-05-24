"""
Entrenador Automático Mejorado - Detección automática de GPU/CPU
"""

import os
import cv2
import torch
from pathlib import Path
from ultralytics import YOLO
import shutil

class ImprovedAutoTrainer:
    def __init__(self):
        self.project_root = Path.cwd()
        self.datasets_dir = self.project_root / "datasets"
        self.training_dir = self.datasets_dir / "yolo_training"
        self.models_dir = self.project_root / "models"
        
        # Detectar GPU/CPU
        self.device = self.detect_device()
        self.training_dir.mkdir(parents=True, exist_ok=True)
        
    def detect_device(self):
        """Detectar GPU o CPU disponible"""
        if torch.cuda.is_available():
            device = "0"  # GPU
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU detectada: {gpu_name}")
            return device
        else:
            print("⚠️  No hay GPU disponible, usando CPU")
            print("   Nota: El entrenamiento será más lento en CPU")
            return "cpu"
    
    def extract_frames_from_video(self, video_path, output_dir, max_frames=150):
        """Extraer frames de un video"""
        print(f"\n  📹 {video_path.name}")
        
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            print(f"     ❌ Error")
            return 0
        
        interval = max(1, total_frames // max_frames)
        frame_count = 0
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % interval == 0:
                frame = cv2.resize(frame, (640, 480))
                frame_path = output_dir / f"{Path(video_path).stem}_frame_{frame_count:04d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                frame_count += 1
            
            frame_idx += 1
        
        cap.release()
        print(f"     ✓ {frame_count} frames")
        return frame_count
    
    def prepare_dataset(self):
        """Preparar dataset"""
        print("\n" + "="*70)
        print("📊 PREPARANDO DATASET")
        print("="*70)
        
        images_dir = self.training_dir / "images"
        labels_dir = self.training_dir / "labels"
        
        for d in [images_dir, labels_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Buscar videos
        videos = [
            v for v in self.datasets_dir.glob("*.mp4")
            if "test" in v.name or "collision" in v.name or "flow" in v.name
        ]
        
        if not videos:
            print("⚠️  Sin videos")
            return False
        
        print(f"\n✓ {len(videos)} video(s) encontrado(s)")
        
        total_frames = 0
        for video in videos:
            frames = self.extract_frames_from_video(video, images_dir, max_frames=100)
            total_frames += frames
        
        print(f"\n✅ Total: {total_frames} frames")
        
        # Crear etiquetas vacías
        for img_file in images_dir.glob("*.jpg"):
            (labels_dir / img_file.with_suffix('.txt').name).write_text("")
        
        # Crear YAML
        yaml_content = f"""path: {self.training_dir.absolute()}
train: images
val: images

nc: 4
names: ['car', 'motorcycle', 'bus', 'truck']
"""
        
        yaml_file = self.training_dir / "data.yaml"
        yaml_file.write_text(yaml_content)
        
        return total_frames > 0
    
    def fine_tune_model(self, epochs=2):
        """Fine-tuning del modelo"""
        print("\n" + "="*70)
        print("🤖 ENTRENAMIENTO DEL MODELO")
        print("="*70)
        
        yaml_path = self.training_dir / "data.yaml"
        
        try:
            print(f"\n📥 Cargando YOLO v8 nano...")
            model = YOLO('yolov8n.pt')
            
            print(f"📊 Configuración:")
            print(f"   Device: {self.device}")
            print(f"   Épocas: {epochs}")
            print(f"   Batch: 2 (optimizado para CPU)")
            print()
            
            # Entrenar
            results = model.train(
                data=str(yaml_path),
                epochs=epochs,
                imgsz=640,
                batch=2,  # Batch pequeño para CPU
                patience=1,
                save=True,
                device=self.device,  # GPU o CPU automáticamente
                verbose=False,
                project='runs/detect',
                name='accident_detection',
                exist_ok=True
            )
            
            print("\n✅ Entrenamiento completado")
            
            # Guardar modelo
            runs_dir = Path("runs/detect/accident_detection")
            best_weights = runs_dir / "weights" / "best.pt"
            
            if best_weights.exists():
                self.models_dir.mkdir(parents=True, exist_ok=True)
                dest = self.models_dir / "yolov8n_fine_tuned.pt"
                shutil.copy(best_weights, dest)
                print(f"💾 Guardado: {dest}")
                return True
            
            return False
        
        except Exception as e:
            print(f"⚠️  {e}")
            return False
    
    def test_model(self):
        """Probar modelo"""
        print("\n" + "="*70)
        print("🧪 PRUEBA DEL MODELO")
        print("="*70)
        
        model_path = self.models_dir / "yolov8n_fine_tuned.pt"
        
        if model_path.exists():
            print(f"\n📥 Usando modelo entrenado")
            model = YOLO(str(model_path))
        else:
            print(f"\n📥 Usando modelo pretrained")
            model = YOLO('yolov8n.pt')
        
        # Probar
        test_video = list(self.datasets_dir.glob("*.mp4"))[0]
        
        print(f"\n🎬 Probando con: {test_video.name}")
        
        results = model.predict(
            source=str(test_video),
            conf=0.5,
            save=True,
            verbose=False
        )
        
        print(f"✓ Inferencia completada")
        print(f"📁 Resultados: runs/detect/predict")
    
    def run(self):
        """Ejecutar pipeline"""
        print("\n")
        print("╔" + "="*68 + "╗")
        print("║" + " "*18 + "🎯 ENTRENADOR AUTOMÁTICO YOLO v8" + " "*16 + "║")
        print("║" + " "*20 + "Con soporte GPU/CPU automático" + " "*18 + "║")
        print("╚" + "="*68 + "╝")
        
        # Paso 1
        if not self.prepare_dataset():
            print("\n❌ Falló preparación")
            return
        
        # Paso 2
        self.fine_tune_model(epochs=2)
        
        # Paso 3
        self.test_model()
        
        # Resumen
        print("\n" + "="*70)
        print("✅ COMPLETADO")
        print("="*70)
        print(f"\n📊 Archivos generados:")
        print(f"   📂 Dataset: {self.training_dir}")
        print(f"   🤖 Modelo: {self.models_dir}")
        print(f"   📁 Resultados: runs/detect/")
        print(f"\n🚀 ¡Streamlit está en http://localhost:8501!")
        print()


if __name__ == "__main__":
    trainer = ImprovedAutoTrainer()
    trainer.run()
