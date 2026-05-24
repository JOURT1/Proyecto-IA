# 🚀 GUÍA DE INICIO RÁPIDO

## Instalación Rápida (5 minutos)

### 1. Instalación de Dependencias

```bash
# Windows
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Ejecutar Streamlit (Interfaz Web)

```bash
streamlit run app/main.py
```

Accede a: **http://localhost:8501**

---

## Primeros Pasos

### Opción A: Interfaz Web Streamlit (Recomendado)

1. Ejecuta `streamlit run app/main.py`
2. Abre http://localhost:8501
3. Carga un video CCTV
4. Ajusta parámetros (opcional)
5. Presiona "Iniciar Análisis"
6. Descarga resultados

### Opción B: Línea de Comandos

```bash
# Procesamiento simple
python app/inference.py --video mi_video.mp4

# Con parámetros personalizados
python app/inference.py \
    --video mi_video.mp4 \
    --confidence 0.55 \
    --export-video \
    --export-json \
    --export-csv
```

Resultados en: `outputs/`

### Opción C: Python Script

```python
from app.processing.pipeline import VideoProcessor

processor = VideoProcessor()
results = processor.process_video(
    "mi_video.mp4",
    output_video_path="salida.mp4",
    export_json=True
)

print(f"Accidentes detectados: {results['accidents_detected']}")
```

---

## Configuración Básica

### Modificar Umbral de Detección

Edita `config/settings.yaml`:

```yaml
model:
  confidence_threshold: 0.5   # Aumenta para menos falsos positivos
  iou_threshold: 0.45         # Aumenta para NMS más agresivo
```

### Ajustar Parámetros de Colisión

```yaml
collision:
  min_signals_for_collision: 3        # 3-5 señales requeridas
  critical_proximity_distance: 1.5    # Distancia crítica en metros
  ignore_slow_collisions_below_velocity: 0.5  # Ignora contacto lento
```

### Parámetros de Severidad

```yaml
severity:
  thresholds:
    leve_max: 30      # 0-30: Leve
    moderado_max: 65  # 31-65: Moderado
    severo_min: 66    # 66+: Severo
```

---

## Videos de Prueba

### Descargar Dataset CADP

```bash
# Dataset de accidentes reales para pruebas
# Disponible en: https://ankitshah009.github.io/accident_forecasting_traffic_camera
```

### Crear Video de Prueba

```bash
# Con OpenCV (Python)
import cv2
import numpy as np

# Crear video de prueba simple
out = cv2.VideoWriter('test.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))
for i in range(300):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame, f'Frame {i}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
    out.write(frame)
out.release()
```

---

## Troubleshooting

### Error: "ModuleNotFoundError: No module named 'ultralytics'"

```bash
pip install ultralytics==8.0.176
```

### Error: "CUDA out of memory"

Edita `config/settings.yaml`:

```yaml
model:
  yolo_model: "yolov8n.pt"  # Usa 'n' (nano) en lugar de 'm' (medium)
  yolo_device: "cpu"         # Cambia a CPU
```

### Video no se procesa

```bash
# Verifica el formato
ffprobe mi_video.mp4

# Convierte a MP4 si es necesario
ffmpeg -i video.avi -c:v libx264 -c:a aac output.mp4
```

### FPS bajo

- Reduce resolución de video
- Usa modelo YOLO nano ("yolov8n.pt")
- Aumenta frame_skip en configuración
- Ejecuta en GPU si es disponible

---

## Estructura de Salida

```
outputs/
├── mi_video_processed.mp4      # Video con anotaciones
├── mi_video_results.json       # Resultados detallados
├── mi_video_events.csv         # Lista de eventos
└── mi_video_summary.json       # Resumen de procesamiento
```

### Contenido JSON

```json
{
  "total_frames": 1500,
  "accidents_detected": 2,
  "fps": 33.2,
  "detected_accidents": [
    {
      "frame": 250,
      "timestamp": "00:08:20",
      "severity": "Severo",
      "score": 87.5
    }
  ]
}
```

### Contenido CSV

```
frame,timestamp,vehicle_ids,severity,score
250,"00:08:20","[1, 5]",Severo,87.5
```

---

## Casos de Uso Comunes

### Procesar Carpeta de Videos

```bash
#!/bin/bash
for video in videos/*.mp4; do
    python app/inference.py --video "$video" --export-json --export-csv
done
```

### Ajustar para Baja Iluminación

En `config/settings.yaml`:

```yaml
model:
  confidence_threshold: 0.40   # Más permisivo
  iou_threshold: 0.40          # Menos NMS
```

### Detectar Solo Choques Severos

```yaml
collision:
  min_signals_for_collision: 4  # Requiere 4 señales
  ignore_slow_collisions_below_velocity: 1.0  # Ignora lentos
```

---

## Rendimiento Esperado

### Especificaciones Mínimas
- **CPU**: Intel i5 / AMD Ryzen 5
- **RAM**: 8 GB
- **Almacenamiento**: 10 GB
- **FPS**: 10-15 FPS en CPU

### Con GPU (Recomendado)
- **GPU**: NVIDIA RTX 3060 o superior
- **VRAM**: 6 GB mínimo
- **FPS**: 30+ FPS en tiempo real

---

## Siguiente Paso: Fine-tuning

Una vez familiarizado, puedes entrenar con dataset propio:

```bash
python app/train.py \
    --data config/dataset.yaml \
    --model yolov8s.pt \
    --epochs 50 \
    --batch-size 16
```

Ver README.md para detalles completos.

---

## Soporte

- 📖 Lee README.md para documentación completa
- 💬 Revisa comentarios en el código
- 🐛 Activa debug mode en configuración
- 📋 Verifica logs en logs/system.log

---

## Próximos Pasos Recomendados

1. ✅ Ejecutar demo con Streamlit
2. ✅ Procesar video de prueba
3. ✅ Explorar parámetros de configuración
4. ✅ Revisar código fuente
5. ✅ Entrenar con dataset propio (opcional)

¡Listo! 🎉
