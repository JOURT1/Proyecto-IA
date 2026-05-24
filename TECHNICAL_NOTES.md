# NOTAS TÉCNICAS - SISTEMA DE DETECCIÓN DE ACCIDENTES

## Contenidos

1. [Arquitectura General](#arquitectura-general)
2. [Flujo de Datos](#flujo-de-datos)
3. [Algoritmos Implementados](#algoritmos-implementados)
4. [Optimizaciones de Rendimiento](#optimizaciones-de-rendimiento)
5. [Casos de Prueba](#casos-de-prueba)
6. [Debugging](#debugging)

---

## Arquitectura General

### Componentes Principales

```
┌─────────────────────────────────────────────────────────────┐
│                    VIDEO INPUT LAYER                         │
│              (OpenCV Video Capture)                          │
└────────────────────────┬────────────────────────────────────┘
                         │ Frames
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                 DETECTION LAYER (YOLO)                       │
│  Input: Frame → Output: List[Detection(bbox, conf, class)] │
│  Model: yolov8n.pt (nano) / yolov8s.pt (small)            │
│  GPU: ✓ (with CUDA) / CPU: ✓                               │
└────────────────────────┬────────────────────────────────────┘
                         │ Detections
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  TRACKING LAYER (BYTETRACK)                 │
│  Input: List[Detection] → Output: List[Track]              │
│  Purpose: Assign persistent IDs across frames              │
│  Memory: O(n_tracks * trajectory_length)                  │
└────────────────────────┬────────────────────────────────────┘
                         │ Tracked Vehicles
                         ▼
┌─────────────────────────────────────────────────────────────┐
│               KINEMATICS LAYER (Analysis)                    │
│  Input: Track → Output: KinematicState                     │
│  Calculations:                                              │
│    - Velocity (pixels/frame → m/s)                         │
│    - Acceleration (m/s²)                                   │
│    - Direction (degrees)                                   │
│    - Proximity (meters)                                    │
└────────────────────────┬────────────────────────────────────┘
                         │ Kinematic States
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              COLLISION DETECTION LAYER                       │
│  Input: Tracks + Kinematics → Output: List[CollisionEvent] │
│  Method: Multi-signal rule engine                          │
│  Signals: proximity, trajectory, velocity, direction       │
│  Threshold: ≥3 signals required                            │
└────────────────────────┬────────────────────────────────────┘
                         │ Collision Events
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              SEVERITY CLASSIFICATION LAYER                   │
│  Input: CollisionEvent + Kinematics                        │
│  Output: SeverityAssessment(level, score, description)     │
│  Scale: 0-100 points                                        │
│    0-30:  Leve (Light)                                     │
│    31-65: Moderado (Moderate)                              │
│    66+:   Severo (Severe)                                  │
└────────────────────────┬────────────────────────────────────┘
                         │ Assessments
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              VISUALIZATION LAYER                             │
│  Input: Frame + Tracks + Collisions + Severity             │
│  Output: Annotated Frame                                   │
│  Features:                                                  │
│    - Bounding boxes                                        │
│    - Track IDs                                             │
│    - Trajectories                                          │
│    - Alert messages                                        │
│    - Progress bar                                          │
└────────────────────────┬────────────────────────────────────┘
                         │ Annotated Frames
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   OUTPUT LAYER                               │
│  - Video file (MP4)                                        │
│  - JSON results                                             │
│  - CSV events                                               │
│  - Summary statistics                                       │
└─────────────────────────────────────────────────────────────┘
```

---

## Flujo de Datos

### Estructura de Datos Principales

#### Detection (de YOLO)
```python
@dataclass
class Detection:
    class_id: int              # 0=car, 1=motorcycle, 2=bus, 3=truck
    class_name: str
    confidence: float          # 0.0-1.0
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    center: Tuple[float, float]      # (cx, cy)
    area: float                # width * height
```

#### Track (de ByteTrack)
```python
@dataclass
class Track:
    track_id: int
    class_name: str
    position_history: deque    # Últimos N centros
    current_bbox: Tuple[int, int, int, int]
    current_center: Tuple[float, float]
    velocity: Tuple[float, float]  # (vx, vy) píxeles/frame
    direction_angle: float     # grados
    speed: float               # píxeles/frame
    age: int                   # frames desde creación
    hits: int                  # frames con detección
```

#### KinematicState
```python
@dataclass
class KinematicState:
    track_id: int
    position: Tuple[float, float]  # metros
    velocity_magnitude: float      # m/s
    velocity_vector: Tuple[float, float]  # (vx, vy)
    direction_angle: float         # grados
    acceleration_magnitude: float  # m/s²
    acceleration_vector: Tuple[float, float]
    velocity_change_rate: float    # m/s por frame
    direction_change_rate: float   # grados por frame
```

#### CollisionEvent
```python
@dataclass
class CollisionEvent:
    frame_number: int
    track_ids: List[int]           # IDs de vehículos
    confidence: float              # 0.0-1.0
    signals: List[str]             # Señales que causaron detección
    distance_at_impact: float      # metros
    relative_velocity: float        # m/s
    vehicles_stopped_after: bool
```

#### SeverityAssessment
```python
@dataclass
class SeverityAssessment:
    level: SeverityLevel           # LEVE / MODERADO / SEVERO
    score: float                   # 0-100
    confidence: float              # 0.0-1.0
    factors: Dict[str, float]      # Scores de cada factor
    description: str               # Explicación textual
```

---

## Algoritmos Implementados

### 1. Detección (YOLO v8)

**Arquitectura:**
- Backbone: CSPDarknet
- Head: Decoder + Detection layers
- Input: 640x640 (adaptable)
- Output: Bounding boxes + confidence

**Características:**
- Detección en tiempo real (~30 ms en GPU)
- NMS (Non-Maximum Suppression) integrado
- Múltiples escalas de objetos

**Parámetros clave:**
```python
confidence_threshold = 0.5  # Filtro de confianza
iou_threshold = 0.45        # Filtro NMS
```

---

### 2. Tracking (ByteTrack)

**Algoritmo:**
```
Para cada frame:
  1. Detectar objetos (YOLO)
  2. Calcular matriz de IoU (detections vs tracked objects)
  3. Matching greedy de alto IoU
  4. Crear tracks para detecciones no matcheadas
  5. Eliminar tracks stale (sin actualización)
```

**Métricas:**
- IoU (Intersection over Union)
- Distancia espacial
- Motion similarity

**Parámetros:**
```python
track_buffer = 30          # Frames antes de eliminar track
match_thresh = 0.8         # IoU mínimo para matching
trajectory_history = 60    # Frames de historia
```

---

### 3. Cinemática

#### Cálculo de Velocidad

```
Velocidad = Δposición / Δtiempo

V(pixels/frame) = (x_actual - x_pasado) / (t_actual - t_pasado)
V(m/s) = V(pixels/frame) * pixel_to_meter * fps

Ejemplo:
  Movimiento: 20 píxeles en 1 frame
  pixel_to_meter: 0.05 m/píxel
  fps: 30 fps
  V = 20 * 0.05 * 30 = 30 m/s
```

#### Cálculo de Aceleración

```
Aceleración = Δvelocidad / Δtiempo

a(m/s²) = (v_actual - v_pasado) * fps

Ejemplo:
  V cambio: 2 m/s en 1 frame
  fps: 30
  a = 2 * 30 = 60 m/s²
```

#### Cálculo de Dirección

```
θ = arctan2(Δy, Δx) * 180/π

Normalizado a [0, 360) grados

Ejemplo:
  Δx=10, Δy=10
  θ = arctan2(10, 10) = 45°
```

#### Cambio de Dirección

```
Δθ = arctan2(v2) - arctan2(v1)

Detecta cambios abruptos de trayectoria

Umbral: > 30° es significativo
```

---

### 4. Detección de Colisiones

**Motor de Reglas:**

```python
def detect_collision(track_a, track_b, kinematics_a, kinematics_b):
    signals = []
    
    # Signal 1: Proximity
    distance = calculate_distance(
        kinematics_a.position,
        kinematics_b.position
    )
    if distance < 1.5:
        signals.append('proximity')
    
    # Signal 2: Trajectory intersection
    if check_trajectories_intersect(track_a, track_b):
        signals.append('trajectory_intersection')
    
    # Signal 3: Velocity change A
    if kinematics_a.velocity_change_rate > 2.0:
        signals.append('velocity_change_a')
    
    # Signal 4: Velocity change B
    if kinematics_b.velocity_change_rate > 2.0:
        signals.append('velocity_change_b')
    
    # Signal 5: Direction change
    if kinematics_a.direction_change_rate > 30:
        signals.append('direction_change_a')
    
    # Signal 6: Permanence after contact
    if movement_after_contact < 0.2:
        signals.append('mutual_stop')
    
    # Require minimum signals
    if len(signals) >= 3:
        return CollisionEvent(
            track_ids=[track_a.id, track_b.id],
            signals=signals,
            confidence=calculate_confidence(signals, distance)
        )
    return None
```

**Detección temporal:**
- Se analiza ventana pre-impacto (15 frames)
- Se analiza ventana post-impacto (20 frames)
- Se requiere confirmación en 3+ frames

---

### 5. Clasificación de Severidad

**Fórmula de puntuación:**

```
score = Σ(factor_i × peso_i) × 100

Donde:
  factor_velocity_change    = |Δv| / 15          (máx 15 m/s)
  factor_direction_change   = |Δθ| / 180         (máx 180°)
  factor_proximity          = 1 - (d / 2)        (máx 2m)
  factor_deceleration       = a / 10              (máx 10 m/s²)
  factor_permanence         = 0.8 si parado, 0.2 si continúa
  factor_trajectory_dist    = n_signals / 6

Pesos (sum = 1.0):
  velocity_change:   0.25
  direction_change:  0.15
  proximity:         0.15
  deceleration:      0.20
  permanence_time:   0.15
  trajectory_dist:   0.10
```

**Clasificación por rango:**
```
Score:      Clasificación:
0-30        Leve (Rozón, contacto menor)
31-65       Moderado (Colisión perceptible)
66-100      Severo (Choque de alto impacto)
```

---

## Optimizaciones de Rendimiento

### 1. Detección

**Técnicas aplicadas:**
- Resize dinámico basado en detecciones
- Batch processing
- Cache de modelos
- Mixed precision (FP16 en GPU)

**Configuración:**
```yaml
target_fps: 15              # Procesar a 15 FPS
frame_skip: 2               # Procesar cada 2do frame
resize_enabled: true
target_height: 480
```

### 2. Tracking

**Optimizaciones:**
- Matriz de IoU vectorizada (NumPy)
- Matching greedy O(n²) vs optimal O(n³)
- Limitación de histórico de posiciones

### 3. Memoria

**Gestión:**
```python
# Limitar historia de posiciones
position_history = deque(maxlen=60)  # Máx 60 frames

# Limitar frames en memoria
max_frames_memory = 300

# Limpiar tracks obsoletos cada frame
```

### 4. Cálculos Cinemáticos

**Vectorización con NumPy:**
```python
# Cálculo de distancias vectorizado
distances = np.sqrt(
    (positions[:, 0:1] - positions[:, 0])[:2] +
    (positions[:, 1:1] - positions[:, 1])**2
)
```

---

## Casos de Prueba

### Test 1: Detección Básica

```python
def test_detection():
    detector = VehicleDetector()
    frame = cv2.imread("test_frame.jpg")
    detections = detector.detect(frame)
    assert len(detections) > 0
    assert all(d.confidence > 0 for d in detections)
```

### Test 2: Tracking Persistencia

```python
def test_tracking_persistence():
    tracker = ByteTracker()
    tracks = tracker.update(detections_frame1)
    ids_frame1 = {t.track_id for t in tracks}
    
    tracks = tracker.update(detections_frame2)
    ids_frame2 = {t.track_id for t in tracks}
    
    # IDs debe manterse consistente
    assert ids_frame1 & ids_frame2  # Intersección no vacía
```

### Test 3: Colisión Detectada

```python
def test_collision_detection():
    detector = CollisionDetector()
    
    # Crear tracks que se aproximan
    track_a.position_history = [(0, 0), (5, 0), (10, 0)]
    track_b.position_history = [(100, 0), (95, 0), (90, 0)]
    
    collision = detector.detect([track_a, track_b], 3, kin_states)
    assert collision is not None
    assert 'proximity' in collision.signals
```

### Test 4: Severidad Leve

```python
def test_severity_light():
    classifier = SeverityClassifier()
    
    collision = CollisionEvent(
        track_ids=[1, 2],
        distance_at_impact=1.8,
        relative_velocity=0.5,
        vehicles_stopped_after=False,
        signals=['proximity']
    )
    
    assessment = classifier.classify(collision, kin_states)
    assert assessment.level == SeverityLevel.LEVE
    assert assessment.score < 30
```

---

## Debugging

### Modo Debug

Edita `config/settings.yaml`:

```yaml
logging:
  level: "DEBUG"
  log_to_console: true
  log_file: "logs/system.log"
```

### Inspeccionar Detecciones

```python
from app.detection.detector import VehicleDetector

detector = VehicleDetector()
frame = cv2.imread("test.jpg")
detections = detector.detect(frame)

for d in detections:
    print(f"ID: {d.class_id}")
    print(f"Class: {d.class_name}")
    print(f"Confidence: {d.confidence:.3f}")
    print(f"BBox: {d.bbox}")
```

### Inspeccionar Tracks

```python
from app.tracking.tracker import ByteTracker

tracker = ByteTracker()
tracks = tracker.update(detections)

for track in tracks:
    if track.is_confirmed():
        print(f"ID: {track.track_id}")
        print(f"Age: {track.age}")
        print(f"Position: {track.current_center}")
        print(f"Velocity: {track.velocity}")
        print(f"Trajectory: {len(list(track.position_history))} points")
```

### Inspeccionar Cinemática

```python
from app.kinematics.calculator import KinematicsCalculator

kinematics = KinematicsCalculator()
kin_state = kinematics.calculate(track)

print(f"Velocity: {kin_state.velocity_magnitude:.2f} m/s")
print(f"Direction: {kin_state.direction_angle:.1f}°")
print(f"Acceleration: {kin_state.acceleration_magnitude:.2f} m/s²")
```

### Inspeccionar Colisiones

```python
from app.collision.detector import CollisionDetector

detector = CollisionDetector(kinematics=kinematics)
collisions = detector.detect(tracks, frame_num, kin_states)

for collision in collisions:
    print(f"Vehicles: {collision.track_ids}")
    print(f"Distance: {collision.distance_at_impact:.2f} m")
    print(f"Signals: {collision.signals}")
    print(f"Confidence: {collision.confidence:.2f}")
```

---

## Recursos Adicionales

- YOLO v8 Docs: https://docs.ultralytics.com/
- ByteTrack Paper: https://arxiv.org/abs/2110.06864
- OpenCV Docs: https://docs.opencv.org/
- PyTorch Docs: https://pytorch.org/docs/

---

**Versión:** 1.0.0
**Última actualización:** 15/04/2026
