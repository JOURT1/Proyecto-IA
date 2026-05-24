# Sistema Inteligente de Detección de Accidentes de Tránsito mediante Visión Artificial

## 📋 Descripción del Proyecto

Sistema completo basado en inteligencia artificial para la detección automática de accidentes de tránsito en videos de vigilancia CCTV urbana. El sistema utiliza visión por computadora avanzada para analizar el comportamiento vehicular, detectar colisiones y clasificar su severidad en tiempo real.

### Características Principales

- **✅ Detección de Vehículos en Tiempo Real** - Basada en YOLO v8
- **✅ Tracking Persistente** - ByteTrack para seguimiento temporal
- **✅ Análisis Cinemático** - Cálculo de velocidad, aceleración y trayectorias
- **✅ Detección de Colisiones** - Motor de reglas multimodal
- **✅ Clasificación de Severidad** - Leve, Moderado, Severo
- **✅ Visualización en Tiempo Real** - Anotaciones y alertas
- **✅ Interfaz Web** - Dashboard interactivo con Streamlit
- **✅ Exportación de Resultados** - JSON, CSV, Video procesado

---

## 🎯 Objetivo del Sistema

Desarrollar un prototipo académico avanzado que demuestre:

1. Aplicación de técnicas de visión artificial moderna
2. Integración de múltiples módulos de IA
3. Análisis cinemático de eventos vehiculares
4. Sistemas de alerta basados en inteligencia
5. Procesamiento eficiente en tiempo real

---

## 🛠️ Stack Tecnológico

### Backend / IA
- **Python 3.11+** - Lenguaje principal
- **PyTorch** - Framework de deep learning
- **Ultralytics YOLO** - Detección de objetos
- **OpenCV** - Procesamiento de video
- **NumPy & Pandas** - Computación científica

### Tracking
- **ByteTrack** - Multi-object tracking

### Interfaz
- **Streamlit** - Dashboard web interactivo

### Gestión
- **Git/GitHub** - Control de versiones
- **YAML** - Configuración

---

## 📁 Estructura del Proyecto

```
Proyecto IA/
├── app/
│   ├── __init__.py
│   ├── main.py                 # Interfaz Streamlit
│   ├── inference.py            # Script de inferencia
│   ├── train.py                # Script de entrenamiento
│   │
│   ├── detection/
│   │   ├── __init__.py
│   │   └── detector.py         # Módulo YOLO
│   │
│   ├── tracking/
│   │   ├── __init__.py
│   │   └── tracker.py          # ByteTrack
│   │
│   ├── kinematics/
│   │   ├── __init__.py
│   │   └── calculator.py       # Cálculos cinemáticos
│   │
│   ├── collision/
│   │   ├── __init__.py
│   │   └── detector.py         # Motor de detección
│   │
│   ├── severity/
│   │   ├── __init__.py
│   │   └── classifier.py       # Clasificador de severidad
│   │
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── visualizer.py       # Anotaciones
│   │
│   ├── processing/
│   │   ├── __init__.py
│   │   └── pipeline.py         # Pipeline integrado
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   └── config.py           # Utilidades
│   │
│   └── exports/
│       └── (módulos de exportación)
│
├── config/
│   └── settings.yaml           # Configuración principal
│
├── models/
│   └── (modelos YOLO)
│
├── datasets/
│   └── (datos de entrenamiento)
│
├── notebooks/
│   └── (Jupyter notebooks)
│
├── outputs/
│   └── (resultados de procesamiento)
│
├── requirements.txt            # Dependencias Python
├── README.md                   # Este archivo
├── .gitignore
└── .github/
    └── workflows/              # CI/CD (opcional)
```

---

## 🚀 Instalación

### Requisitos Previos

- Python 3.11 o superior
- pip (gestor de paquetes Python)
- 4 GB de RAM mínimo (8+ GB recomendado)
- GPU NVIDIA (opcional, pero recomendada)

### Pasos de Instalación

1. **Clonar o descargar el proyecto:**
   ```bash
   cd "Proyecto IA"
   ```

2. **Crear entorno virtual (recomendado):**
   ```bash
   # En Windows
   python -m venv venv
   venv\Scripts\activate
   
   # En macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Descargar modelo YOLO (automático en primera ejecución):**
   - El sistema descargará automáticamente los pesos de YOLO en primera ejecución
   - La descarga se hace una sola vez y se cachea

---

## 📊 Uso del Sistema

### 1. Interfaz Streamlit (Recomendado para Demostración)

```bash
streamlit run app/main.py
```

**Funcionalidades:**
- Upload de video CCTV
- Ajuste de parámetros de detección
- Procesamiento en tiempo real
- Visualización de resultados
- Descarga de video procesado y reportes

**Acceso:** http://localhost:8501

### 2. Inferencia por Línea de Comandos

```bash
# Procesamiento básico
python app/inference.py --video datasets/test/video.mp4

# Con parámetros personalizados
python app/inference.py \
    --video datasets/test/video.mp4 \
    --confidence 0.5 \
    --iou 0.45 \
    --export-video \
    --export-json \
    --export-csv
```

**Parámetros disponibles:**
- `--video`: Ruta del video de entrada (obligatorio)
- `--output`: Directorio de salida (default: outputs/)
- `--confidence`: Umbral de confianza YOLO (default: 0.5)
- `--iou`: Umbral IoU para NMS (default: 0.45)
- `--config`: Ruta de configuración (default: config/settings.yaml)
- `--export-video`: Exportar video procesado
- `--export-json`: Exportar resultados JSON
- `--export-csv`: Exportar eventos CSV

### 3. Entrenamiento Fine-tuning

```bash
# Fine-tuning con dataset propio
python app/train.py \
    --data config/dataset.yaml \
    --model yolov8s.pt \
    --epochs 50 \
    --batch-size 16 \
    --device 0
```

**Nota:** Requiere dataset con estructura específica. Ver [Configuración de Dataset](#configuración-de-dataset)

---

## ⚙️ Configuración

### Archivo Principal: `config/settings.yaml`

El sistema se configura completamente sin modificar código:

```yaml
# Detección
model:
  yolo_model: "yolov8n.pt"      # Modelo YOLO
  confidence_threshold: 0.50     # Umbral de confianza
  vehicle_classes: [car, motorcycle, bus, truck]

# Tracking
tracking:
  track_buffer: 30               # Frames para mantener track
  match_thresh: 0.8              # Umbral de matching

# Cinemática
kinematics:
  fps: 30                        # Frames por segundo
  pixel_to_meter: 0.05           # Conversión de píxeles a metros
  velocity_change_threshold: 2.0 # m/s

# Detección de Colisiones
collision:
  min_signals_for_collision: 3   # Señales mínimas requeridas
  collision_confirmation_frames: 3

# Severidad
severity:
  thresholds:
    leve_max: 30
    moderado_max: 65
    severo_min: 66

# Exportación
export:
  output_dir: "outputs/"
  export_video: true
  export_json: true
  export_csv: true
```

**Modifica estos valores sin necesidad de cambiar código fuente.**

---

## 📚 Arquitectura del Sistema

### Pipeline de Procesamiento

```
Video Input
    ↓
[Frame Extraction]
    ↓
[Detection] → YOLO detecta vehículos
    ↓
[Tracking] → ByteTrack asigna IDs persistentes
    ↓
[Kinematics] → Cálcula velocidad, aceleración, trayectorias
    ↓
[Collision Detection] → Motor de reglas multimodal
    ↓
[Severity Classification] → Clasifica en Leve/Moderado/Severo
    ↓
[Visualization] → Dibuja anotaciones y alertas
    ↓
[Output] → Video + JSON + CSV
```

### Módulos Principales

#### 1. **Detection** (`app/detection/detector.py`)
- Usa YOLOv8 para detección de vehículos
- Detecta: automóvil, motocicleta, bus, camión
- Salida: bounding boxes, confianza, centros

#### 2. **Tracking** (`app/tracking/tracker.py`)
- ByteTrack para tracking persistente
- Mantiene ID de vehículos entre frames
- Reconstruye trayectorias

#### 3. **Kinematics** (`app/kinematics/calculator.py`)
- Calcula velocidad (m/s)
- Calcula aceleración (m/s²)
- Calcula dirección (grados)
- Detecta cambios bruscos

#### 4. **Collision** (`app/collision/detector.py`)
Motor de reglas que analiza:
- Proximidad entre vehículos
- Intersección de trayectorias
- Cambios de velocidad
- Cambios de dirección
- Permanencia post-impacto

**Requiere mínimo 3 señales combinadas para detectar colisión**

#### 5. **Severity** (`app/severity/classifier.py`)
Clasificación basada en:
- Cambio de velocidad
- Cambio de dirección
- Proximidad
- Desaceleración
- Tiempo de permanencia

**Rango de puntuación: 0-100**
- 0-30: Leve
- 31-65: Moderado
- 66-100: Severo

#### 6. **Visualization** (`app/visualization/visualizer.py`)
- Bounding boxes de detecciones
- Trayectorias (trails)
- IDs de vehículos
- Alertas por severidad
- Barra de progreso

---

## 🎓 Ejemplos de Uso

### Ejemplo 1: Procesamiento Simple

```python
from app.processing.pipeline import VideoProcessor

# Inicializar
processor = VideoProcessor()

# Procesar video
results = processor.process_video(
    "path/to/video.mp4",
    output_video_path="output.mp4",
    visualize=True,
    export_json=True,
    export_csv=True
)

# Ver resultados
print(f"Accidentes detectados: {results['accidents_detected']}")
for accident in results['detected_accidents']:
    print(f"  - Frame {accident['frame']}: {accident['severity']}")
```

### Ejemplo 2: Procesamiento Frame por Frame

```python
from app.processing.pipeline import VideoProcessor
import cv2

processor = VideoProcessor()

# Abrir video
cap = cv2.VideoCapture("video.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Procesar frame
    results = processor.process_frame(frame)
    
    # Acceder a detecciones, tracking, colisiones, etc.
    collisions = results['collisions']
    severity = results['severity']
    
cap.release()
```

### Ejemplo 3: Configuración Personalizada

```python
from app.utils.config import Config
from app.processing.pipeline import VideoProcessor

# Cargar y modificar configuración
config = Config()
config.load_config("config/settings.yaml")
config.set("model.confidence_threshold", 0.6)
config.set("collision.min_signals_for_collision", 4)

# Crear procesador con configuración personalizada
processor = VideoProcessor()
results = processor.process_video("video.mp4")
```

---

## 📈 Métricas y Evaluación

### Métricas de Desempeño

El sistema reporta automáticamente:

- **FPS**: Frames procesados por segundo
- **Tiempo de inferencia**: Promedio por frame
- **Accuracy de tracking**: Basado en IoU
- **Falsos positivos**: Colisiones erróneas
- **Cobertura**: % de accidentes detectados

### Archivo de Resultados JSON

```json
{
  "status": "success",
  "input_video": "video.mp4",
  "total_frames": 1500,
  "total_time_seconds": 45.2,
  "fps": 33.2,
  "accidents_detected": 3,
  "detected_accidents": [
    {
      "frame": 250,
      "timestamp": "00:08:20",
      "vehicle_ids": [1, 5],
      "severity": "Severo",
      "score": 87.5,
      "description": "..."
    }
  ],
  "metrics": {
    "avg_detection_time_ms": 12.5,
    "avg_tracking_time_ms": 3.2,
    "avg_collision_detection_time_ms": 5.1
  }
}
```

### Archivo CSV de Eventos

```csv
frame,timestamp,vehicle_ids,severity,score,confidence
250,"00:08:20","[1, 5]",Severo,87.5,0.95
580,"00:19:20","[2, 8]",Moderado,55.3,0.82
```

---

## 🔧 Configuración de Dataset

### Estructura Requerida para Fine-tuning

```
dataset/
├── images/
│   ├── train/      # 70% de imágenes
│   ├── val/        # 15% de imágenes
│   └── test/       # 15% de imágenes
│
└── labels/         # Anotaciones YOLO (formato TXT)
    ├── train/
    ├── val/
    └── test/
```

### Archivo `config/dataset.yaml`

```yaml
path: /path/to/dataset
train: images/train
val: images/val
test: images/test

nc: 4
names:
  0: car
  1: motorcycle
  2: bus
  3: truck
```

### Formatos de Anotación Soportados

- **YOLO TXT**: x_center y_center width height (normalizado)
- **COCO JSON**: Objeto JSON con bounding boxes
- **Pascal VOC XML**: XML con xmin, ymin, xmax, ymax

---

## 🌍 Datasets Disponibles

### CADP (Car Accident Detection and Prediction)
- **Fuente**: Público (Academia)
- **Videos**: 1,416 segmentos
- **Descripción**: Cámaras CCTV de tráfico urbano
- **Anotaciones**: Espaciotemporales

### ACCIDENT
- **Fuente**: arXiv (2604.09819)
- **Descripción**: Dataset moderno de accidentes vehiculares
- **Ventaja**: Anotaciones detalladas

### DoTA
- **Fuente**: GitHub (MoonBlvd)
- **Descripción**: Detección de anomalías en tráfico
- **Uso**: Validación comparativa

---

## 📖 Cálculos Cinemáticos Implementados

### Velocidad (m/s)
```
v = Δposition / Δtime
```
- Se calcula sobre una ventana de frames configurable
- Se suaviza para reducir ruido

### Aceleración (m/s²)
```
a = Δv / Δtime
```
- Cambio de velocidad por frame
- Se acumula en historial para análisis temporal

### Dirección (grados)
```
θ = atan2(Δy, Δx) * 180/π
```
- Ángulo de movimiento del vehículo
- Rango: 0-360 grados

### Proximidad (metros)
```
d = √[(x₂-x₁)² + (y₂-y₁)²] * pixel_to_meter
```
- Distancia Euclidiana entre centroides
- Configurable: crítica < 1.5 m

---

## 🚨 Lógica de Detección de Colisiones

### Motor de Reglas Multimodal

Una colisión se detecta si se cumplen **mínimo 3 de las siguientes señales**:

1. **Proximidad Anómala**
   - Distancia < 1.5 metros

2. **Intersección de Trayectorias**
   - Las rutas de vehículos se cruzan

3. **Cambio Brusco de Velocidad (Vehículo A)**
   - Δv > 2.0 m/s en 3 frames

4. **Cambio Brusco de Velocidad (Vehículo B)**
   - Δv > 2.0 m/s en 3 frames

5. **Cambio de Dirección**
   - Δθ > 30° en un frame

6. **Permanencia Post-impacto**
   - Ambos vehículos detienen < 0.2 m en 2 frames

### Reducción de Falsos Positivos

- Ignora contactos a baja velocidad (< 0.5 m/s)
- Requiere confirmación en múltiples frames
- Filtra duplicados en el mismo instante
- Valida coherencia temporal

---

## 📊 Severidad: Sistema de Puntuación

### Factores Considerados (pesos)

- **Cambio de Velocidad** (25%)
  - Rango: 0-15 m/s
  - Score: cambio / 15

- **Cambio de Dirección** (15%)
  - Rango: 0-180°
  - Score: ángulo / 180

- **Proximidad** (15%)
  - Rango: 0-2 m
  - Score: 1 - (distancia / 2)

- **Desaceleración** (20%)
  - Rango: 0-10 m/s²
  - Score: aceleración / 10

- **Permanencia** (15%)
  - Si se detiene post-impacto: 0.8
  - Si continúa: 0.2

- **Distorsión de Trayectoria** (10%)
  - Número de señales / total posible

### Cálculo Final

```
score = Σ(factor_i × peso_i) × 100

Clasificación:
- 0-30: Leve (rozón, contacto menor)
- 31-65: Moderado (colisión perceptible)
- 66-100: Severo (choque alto impacto)
```

---

## 🎯 Casos de Uso

### 1. Monitoreo de Intersecciones
- Vigilancia 24/7 de cruces urbanos
- Alertas automáticas en tiempo real
- Historial de incidentes

### 2. Análisis Forense
- Revisión automática de videos
- Identificación de responsables
- Reconstrucción de eventos

### 3. Mejora de Seguridad Vial
- Identificación de puntos críticos
- Análisis de patrones de accidentes
- Toma de decisiones basada en datos

### 4. Investigación Académica
- Análisis cinemático de vehículos
- Detección de anomalías
- Modelado de comportamiento

---

## ⚠️ Limitaciones Conocidas

### Técnicas

1. **Velocidad Estimada**
   - Aproximación sin calibración de cámara real
   - Válida solo para comparativas relativas

2. **Severidad Estimada**
   - Basada en cinemática, no en física real
   - No mide daño material actual

3. **Baja Iluminación**
   - Degradación de YOLO en luz extrema
   - Solución: mejorar iluminación o usar IR

4. **Oclusión Severa**
   - Pérdida de tracking si objeto se oculta
   - Solución: usar múltiples cámaras

5. **Cámaras Móviles**
   - Sistema diseñado para cámaras fijas
   - No soporta dashcams

### Operacionales

- Requiere calibración por cámara
- Parámetros ajustables pero no automáticos
- Prototipo académico, no certificado
- No sustituye sistemas de emergencia reales

---

## 🔄 Mejoras Futuras

### Corto Plazo
- [ ] Soporte para múltiples cámaras
- [ ] Calibración automática de cámara
- [ ] Detección de peatones/motociclistas
- [ ] Almacenamiento en BD

### Mediano Plazo
- [ ] Modelos 3D de vehículos
- [ ] Estimación de daño material
- [ ] Integración con sistemas de tráfico
- [ ] API REST para terceros

### Largo Plazo
- [ ] Deep learning para severidad
- [ ] Predicción de accidentes
- [ ] Sistema descentralizado
- [ ] Certificación para uso real

---

## 🤝 Contribuciones

Este es un proyecto académico. Las contribuciones son bienvenidas:

1. Mejoras en detección
2. Optimización de rendimiento
3. Nuevos datasets
4. Documentación
5. Ejemplos de uso

---

## 📝 Licencia

Este proyecto es académico y educacional. Uso libre para propósitos académicos.

---

## 👥 Autores

- **Jhoel Alexander Suárez Santander**
- **Mauricio Mora**

**Profesor:** Enrique Vinicio Carrera

**Institución:** UDLA - Carrera de Ingeniería de Software

**Asignatura:** Inteligencia Artificial 1

**Fecha:** 15/04/2026

---

## 📞 Soporte

Para preguntas o problemas:

1. Revisar esta documentación
2. Consultar comentarios en el código
3. Ejecutar con modo debug
4. Revisar logs en `logs/system.log`

---

## 🙏 Agradecimientos

- **Ultralytics** - YOLO v8
- **ByteTrack** - Tracking algorithm
- **OpenCV** - Computer vision
- **Streamlit** - Web framework
- **PyTorch** - Deep learning framework

---

**Última actualización:** 15/04/2026
**Versión:** 1.0.0
