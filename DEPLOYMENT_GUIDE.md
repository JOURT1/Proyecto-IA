# 🚀 GUÍA DE DESPLIEGUE Y VALIDACIÓN

## Checklist de Verificación Previa

### 1. Verificar Python Instalado
```bash
python --version  # Debe ser 3.11+
pip --version
```

### 2. Verificar Estructura de Carpetas
```
Proyecto IA/
├── app/
│   ├── __init__.py ✓
│   ├── main.py ✓
│   ├── inference.py ✓
│   ├── train.py ✓
│   ├── detection/ ✓
│   ├── tracking/ ✓
│   ├── kinematics/ ✓
│   ├── collision/ ✓
│   ├── severity/ ✓
│   ├── visualization/ ✓
│   ├── processing/ ✓
│   └── utils/ ✓
├── config/
│   ├── settings.yaml ✓
│   └── dataset.yaml ✓
├── requirements.txt ✓
├── README.md ✓
├── QUICKSTART.md ✓
├── TECHNICAL_NOTES.md ✓
├── PROJECT_SUMMARY.md ✓
└── examples.py ✓
```

---

## Instalación Paso a Paso

### Paso 1: Crear Entorno Virtual

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

Verificar: Debe mostrar `(venv)` en el prompt

### Paso 2: Actualizar pip

```bash
python -m pip install --upgrade pip
```

### Paso 3: Instalar Dependencias

```bash
pip install -r requirements.txt
```

Esperar a que complete (~5-10 minutos)

### Paso 4: Verificar Instalación

```bash
python -c "import torch; print(torch.__version__)"
python -c "import cv2; print(cv2.__version__)"
python -c "from ultralytics import YOLO; print('YOLO OK')"
python -c "import streamlit; print('Streamlit OK')"
```

Todos deben ejecutarse sin errores.

---

## Prueba de Funcionalidad

### Test 1: Importar Módulos

```bash
python -c "from app.detection import VehicleDetector; print('Detection OK')"
python -c "from app.tracking import ByteTracker; print('Tracking OK')"
python -c "from app.kinematics import KinematicsCalculator; print('Kinematics OK')"
python -c "from app.collision import CollisionDetector; print('Collision OK')"
python -c "from app.severity import SeverityClassifier; print('Severity OK')"
python -c "from app.visualization import FrameVisualizer; print('Visualization OK')"
python -c "from app.processing import VideoProcessor; print('Processing OK')"
```

**Resultado esperado:** Todos los prints sin errores

### Test 2: Cargar Configuración

```bash
python -c "
from app.utils.config import Config
config = Config()
config.load_config('config/settings.yaml')
print('Confidence threshold:', config.get('model.confidence_threshold'))
print('Configuration OK')
"
```

**Resultado esperado:**
```
Confidence threshold: 0.5
Configuration OK
```

### Test 3: Crear Detector YOLO

```bash
python -c "
from app.detection import VehicleDetector
import cv2
import numpy as np

detector = VehicleDetector()
frame = np.zeros((480, 640, 3), dtype=np.uint8)
detections = detector.detect(frame)
print(f'Detector working: {len(detections)} detections in empty frame')
"
```

**Resultado esperado:**
```
Detector working: 0 detections in empty frame
```

### Test 4: Ejecutar Streamlit

```bash
streamlit run app/main.py
```

**Resultado esperado:**
- Abre navegador en http://localhost:8501
- Interfaz carga correctamente
- Sin errores en terminal

Presionar `Ctrl+C` para detener

---

## Validación Completa

### Paso 1: Verificar Instalación de GPU (Opcional)

```bash
python -c "
import torch
print('CUDA available:', torch.cuda.is_available())
print('CUDA device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')
print('PyTorch version:', torch.__version__)
"
```

Si CUDA está disponible, procesamiento será **30+ FPS**
Si solo CPU, procesamiento será **10-15 FPS**

### Paso 2: Crear Video de Prueba

```bash
python -c "
import cv2
import numpy as np
import os

os.makedirs('datasets/test', exist_ok=True)

# Crear video de prueba simple
out = cv2.VideoWriter(
    'datasets/test/test_video.mp4',
    cv2.VideoWriter_fourcc(*'mp4v'),
    30,
    (640, 480)
)

# Generar 300 frames (10 segundos a 30 FPS)
for i in range(300):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame, f'Frame {i}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    out.write(frame)

out.release()
print('Test video created: datasets/test/test_video.mp4')
"
```

### Paso 3: Procesar Video de Prueba

```bash
python app/inference.py \
    --video datasets/test/test_video.mp4 \
    --output outputs/test_run \
    --export-json \
    --export-csv
```

**Resultado esperado:**
- Se procesan 300 frames
- Se genera video procesado
- Se crea outputs/test_run/test_video_results.json
- Se crea outputs/test_run/test_video_events.csv
- No hay accidentes detectados (video vacío)

### Paso 4: Verificar Salidas

```bash
# Verificar archivos generados
ls -la outputs/test_run/

# Ver contenido JSON
python -c "
import json
with open('outputs/test_run/test_video_results.json') as f:
    results = json.load(f)
    print('FPS:', results['fps'])
    print('Total frames:', results['total_frames'])
    print('Time:', results['total_time_seconds'])
"
```

---

## Troubleshooting

### Problema: "ModuleNotFoundError"

**Causa:** Dependencias no instaladas

**Solución:**
```bash
pip install -r requirements.txt --upgrade
```

### Problema: "CUDA out of memory"

**Causa:** GPU insuficiente para modelo

**Solución:**
```bash
# Usar modelo más pequeño
# En config/settings.yaml:
# model:
#   yolo_model: "yolov8n.pt"  # Cambiar a nano
```

O usar CPU:
```bash
# En config/settings.yaml:
# model:
#   yolo_device: "cpu"
```

### Problema: "Video codec not found"

**Causa:** Codec MP4 no disponible

**Solución (Windows):**
```bash
# Instalar ffmpeg
pip install opencv-python-headless
```

**Solución (macOS):**
```bash
brew install ffmpeg
```

**Solución (Linux):**
```bash
sudo apt-get install ffmpeg
```

### Problema: "Streamlit port already in use"

**Causa:** Puerto 8501 ocupado

**Solución:**
```bash
streamlit run app/main.py --server.port 8502
```

### Problema: Bajo rendimiento en CPU

**Causa:** Procesamiento lento en CPU

**Solución:**
```yaml
# config/settings.yaml
processing:
  frame_skip: 2           # Procesar cada 2do frame
  target_height: 360      # Reducir resolución
```

---

## Verificación de Rendimiento

### Test de Velocidad en GPU

```bash
python -c "
from app.processing.pipeline import VideoProcessor
from app.utils import PerformanceMetrics
import time

processor = VideoProcessor()

# Crear frame de prueba
import numpy as np
frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

# Calentar
for _ in range(5):
    processor.process_frame(frame)

# Medir
start = time.time()
for _ in range(30):
    processor.process_frame(frame)
elapsed = time.time() - start

fps = 30 / elapsed
print(f'Performance: {fps:.1f} FPS')
"
```

**Valores esperados:**
- GPU: 30+ FPS
- CPU: 10-15 FPS

---

## Verificación de Exactitud

### Test de Detección

```python
# test_detection.py
from app.detection import VehicleDetector
import cv2

detector = VehicleDetector()

# Cargar imagen con vehículos
# (Se requiere imagen real con vehículos)
frame = cv2.imread("test_image.jpg")

if frame is not None:
    detections = detector.detect(frame)
    print(f"Detections: {len(detections)}")
    for d in detections:
        print(f"  - {d.class_name}: {d.confidence:.2f}")
else:
    print("Use your own image with vehicles for testing")
```

---

## Deployment en Servidor

### Opción 1: Streamlit Cloud

```bash
# 1. Subir proyecto a GitHub
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/username/repo.git
git push -u origin main

# 2. Ir a https://streamlit.io/cloud
# 3. Conectar GitHub y seleccionar repo
# 4. Configurar main.py como aplicación
```

### Opción 2: Servidor Local (Desarrollo)

```bash
# Terminal 1: Ejecutar backend
python app/inference.py --video input.mp4

# Terminal 2: Ejecutar Streamlit
streamlit run app/main.py
```

### Opción 3: Docker (Producción)

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
# Construir y ejecutar
docker build -t accident-detection .
docker run -p 8501:8501 accident-detection
```

---

## Checklist Final

- [ ] Python 3.11+ instalado
- [ ] Entorno virtual activado
- [ ] requirements.txt instalado
- [ ] Todos los módulos importan correctamente
- [ ] Configuración se carga
- [ ] YOLO descargado (primera ejecución)
- [ ] Video de prueba procesa
- [ ] Streamlit ejecuta sin errores
- [ ] Salidas se generan correctamente
- [ ] Rendimiento es aceptable

---

## Verificación del Sistema Completo

### Script de Verificación Automática

```python
# verify_system.py
import sys
import subprocess

checks = [
    ("Python version", "python --version"),
    ("pip", "pip --version"),
    ("PyTorch", "python -c \"import torch; print(torch.__version__)\""),
    ("OpenCV", "python -c \"import cv2; print(cv2.__version__)\""),
    ("YOLO", "python -c \"from ultralytics import YOLO; print('OK')\""),
    ("Streamlit", "python -c \"import streamlit; print('OK')\""),
]

print("System Verification")
print("=" * 50)

all_ok = True
for name, cmd in checks:
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ {name}: OK")
        else:
            print(f"✗ {name}: FAILED")
            all_ok = False
    except Exception as e:
        print(f"✗ {name}: ERROR - {e}")
        all_ok = False

print("=" * 50)
if all_ok:
    print("✅ Sistema listo para usar")
else:
    print("❌ Hay problemas. Ver detalles arriba")

sys.exit(0 if all_ok else 1)
```

```bash
python verify_system.py
```

---

## Contacto y Soporte

**Problemas comunes:** Ver README.md

**Problemas técnicos:** Ver TECHNICAL_NOTES.md

**Ejemplos de uso:** Ver examples.py

**Guía rápida:** Ver QUICKSTART.md

---

**Última actualización:** 15/04/2026
**Versión:** 1.0.0
