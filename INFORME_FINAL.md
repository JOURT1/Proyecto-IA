# Sistema Inteligente de Detección de Accidentes de Tránsito mediante Visión Artificial

## Encabezado

**Asignatura:** Inteligencia Artificial 1  
**Carrera:** Ingeniería de Software - UDLA  
**Integrantes:** Jhoel Alexander Suárez Santander y Mauricio Mora  
**Profesor:** Enrique Vinicio Carrera  
**Fecha de elaboración:** 09/07/2026  
**Versión final del prototipo:** 2.0.0  
**Repositorio:** https://github.com/JOURT1/Proyecto-IA

## Resumen

Este proyecto desarrolló y mejoró un sistema inteligente para detectar accidentes de tránsito en videos de vigilancia urbana mediante visión artificial. El prototipo integra detección de vehículos con YOLOv8, seguimiento temporal, análisis cinemático, detección de colisiones, clasificación de severidad y exportación de resultados en video, JSON y CSV. A partir del prototipo inicial se implementaron mejoras orientadas a eficiencia y eficacia: procesamiento con redimensionamiento y salto configurable de frames, tracking más robusto ante oclusiones, reinicio seguro de estados entre videos, exportación consistente de resultados y una clasificación de severidad basada en señales cinemáticas más informativas. La evaluación final contrasta el sistema antes y después de las mejoras, mostrando un prototipo más estable, rápido y preparado para experimentación académica.

## Introducción

Los accidentes de tránsito representan un problema relevante para la seguridad vial y la gestión urbana. En muchas ciudades existen cámaras CCTV instaladas en intersecciones y vías principales, pero la supervisión manual de estos videos es costosa, lenta y propensa a errores humanos, especialmente cuando se deben monitorear múltiples cámaras al mismo tiempo. Por esta razón, el uso de inteligencia artificial y visión por computador permite construir herramientas de apoyo capaces de analizar video, identificar vehículos, estimar su comportamiento y generar alertas ante eventos compatibles con accidentes.

El problema abordado consiste en detectar automáticamente posibles colisiones vehiculares a partir de videos de vigilancia. Para resolverlo, se implementó una tubería de procesamiento que recibe un video, detecta vehículos, mantiene sus identidades a lo largo del tiempo, calcula variables de movimiento y evalúa señales de accidente como proximidad extrema, convergencia de trayectorias, contacto físico, desaceleración brusca y cambios de dirección. Finalmente, el sistema clasifica la severidad del evento y exporta los resultados para análisis posterior.

La propuesta final conserva el enfoque modular de la fase anterior, pero incorpora mejoras identificadas durante la experimentación preliminar. Estas mejoras buscan reducir falsos positivos, aumentar la estabilidad del tracking, mejorar el rendimiento por frame y asegurar que los resultados puedan reproducirse y analizarse de forma estructurada.

## Implementación

El sistema fue implementado en Python con una arquitectura modular. La detección de vehículos se realiza mediante Ultralytics YOLOv8, el procesamiento de video se realiza con OpenCV, los cálculos numéricos con NumPy y la interfaz de demostración con Streamlit. El flujo general del sistema es el siguiente:

```text
Video de entrada
  -> Captura y preprocesamiento
  -> Detección de vehículos con YOLOv8
  -> Tracking persistente
  -> Análisis cinemático
  -> Detección de colisiones
  -> Clasificación de severidad
  -> Visualización del video procesado
  -> Exportación JSON/CSV/MP4
```

La estructura principal del proyecto es:

```text
app/
  detection/       Detector YOLOv8
  tracking/        Seguimiento de vehículos
  kinematics/      Velocidad, aceleración, dirección y distancia
  collision/       Motor de detección de colisiones
  severity/        Clasificación de severidad
  visualization/   Anotación visual de resultados
  processing/      Pipeline integrado
  utils/           Configuración y métricas
config/
  settings.yaml    Parámetros del sistema
datasets/          Datos y videos de prueba
outputs/           Resultados exportados
```

### Mejoras implementadas respecto al prototipo anterior

**1. Optimización de rendimiento.**  
El prototipo final utiliza los parámetros `processing.frame_skip`, `processing.resize_enabled` y `processing.target_height` definidos en `config/settings.yaml`. En la versión anterior estos valores existían en configuración, pero no eran aplicados por el pipeline. Ahora el sistema puede procesar una versión redimensionada del video y saltar frames de forma controlada, reduciendo el tiempo de inferencia por video. El FPS efectivo se ajusta internamente para que las estimaciones de velocidad y aceleración se mantengan coherentes.

**2. Tracking más robusto ante oclusiones.**  
El tracker inicial asociaba detecciones principalmente mediante IoU entre cajas delimitadoras. La versión final incorpora predicción de posición, distancia entre centroides, consistencia de clase y confianza de detección. Esta mejora permite mantener IDs más estables cuando un vehículo se mueve rápido, cuando hay pérdida breve de detección o cuando dos vehículos se cruzan parcialmente.

**3. Reinicio seguro del estado entre ejecuciones.**  
Como la interfaz Streamlit puede mantener objetos en caché, el sistema ahora reinicia el tracker, los historiales cinemáticos y los estados de colisión al iniciar cada video. Esto evita que resultados de un análisis anterior contaminen el siguiente procesamiento.

**4. Clasificación de severidad mejorada.**  
La severidad dejó de usar un valor fijo como aproximación de energía y ahora considera velocidad relativa del impacto y aceleración máxima de los vehículos involucrados. Con esto, la clasificación leve, moderada o severa se relaciona mejor con señales físicas observables.

**5. Exportación más confiable.**  
La inferencia por consola ahora respeta el directorio definido con `--output` para JSON y CSV. Además, el CSV se genera incluso cuando no se detectan accidentes, dejando encabezados consistentes para análisis posterior.

**6. Inferencia más limpia.**  
La detección YOLO usa `verbose=False` e incorpora `model.image_size` como parámetro configurable. Esto reduce ruido en consola y permite ajustar el tamaño de inferencia sin tocar el código.

## Evaluación

La evaluación compara el desempeño reportado en el prototipo de la fase anterior con las mejoras implementadas en la versión final. Las métricas base del informe 2 fueron:

| Componente | Métrica principal | Resultado fase 2 |
|---|---:|---:|
| Detección de vehículos | F1-score | 0.90 |
| Tracking | MOTA | 0.85 |
| Tracking | ID switches | 12 |
| Detección de colisiones | F1-score | 0.93 |
| Severidad leve | F1-score | 0.86 |
| Severidad moderada | F1-score | 0.81 |
| Severidad severa | F1-score | 0.94 |
| Rendimiento | FPS aproximado | 11 FPS |

Después de las mejoras, la comparación esperada del sistema es:

| Aspecto evaluado | Antes | Después de mejoras | Impacto |
|---|---:|---:|---|
| Uso de `frame_skip` y redimensionamiento | No aplicado | Aplicado en pipeline | Mayor eficiencia |
| FPS esperado en videos 720p/1080p | 8-11 FPS | 12-18 FPS según hardware y escena | Mejora de rendimiento |
| Estabilidad de IDs | Asociación IoU simple | Score combinado con predicción | Menos cambios de ID |
| Procesamiento repetido en Streamlit | Riesgo de estado acumulado | Estado reiniciado por video | Mayor confiabilidad |
| Exportación CSV sin accidentes | No siempre generaba archivo | Archivo con encabezados | Mejor trazabilidad |
| Severidad | Energía aproximada fija | Energía por velocidad relativa y aceleración | Clasificación más coherente |

El análisis cualitativo muestra que las mejoras atacan directamente las limitaciones detectadas en la fase anterior. La optimización del pipeline reduce la carga computacional del detector, que era el principal cuello de botella. El tracking mejorado fortalece la continuidad temporal, lo cual beneficia tanto el cálculo de velocidad como la detección de colisiones. La clasificación de severidad ahora depende de señales cinemáticas del evento, por lo que diferencia mejor entre contactos leves y eventos de mayor impacto.

En términos de eficacia, se espera una reducción de falsos positivos en escenas donde los vehículos están cerca pero no colisionan, ya que el sistema mantiene señales múltiples: proximidad, convergencia, contacto físico, desaceleración y cambios de dirección. En términos de eficiencia, el redimensionamiento y el salto de frames permiten procesar videos más pesados en menor tiempo, manteniendo una salida visual y estructurada útil para revisión humana.

Para reproducir una evaluación local se recomienda ejecutar:

```bash
python generate_test_videos.py
python app/inference.py --video datasets/collision_test.mp4 --output outputs/evaluacion_final --confidence 0.20 --iou 0.45
python app/inference.py --video datasets/traffic_flow.mp4 --output outputs/evaluacion_final --confidence 0.20 --iou 0.45
```

Los archivos resultantes permiten revisar el video anotado, el resumen JSON y la tabla CSV de eventos. Para una validación cuantitativa más rigurosa se debe usar un conjunto etiquetado manualmente con accidentes/no accidentes y severidad real.

## Conclusiones

El proyecto permitió construir un sistema funcional de detección de accidentes de tránsito mediante visión artificial, integrando detección de objetos, tracking, análisis cinemático, reglas de colisión, severidad y exportación de resultados. La versión final mejora el prototipo inicial al hacerlo más eficiente, más estable en ejecuciones repetidas y más coherente en la evaluación de severidad.

Una lección importante es que la detección de accidentes no depende únicamente de reconocer vehículos o medir proximidad. Para evitar falsos positivos se requiere analizar el comportamiento temporal: trayectorias, velocidad relativa, desaceleración, dirección y permanencia posterior al impacto. También se observó que el tracking es un componente crítico, porque un cambio incorrecto de ID puede afectar todo el análisis cinemático.

Como trabajo futuro, el sistema podría incorporar calibración por homografía para convertir píxeles a metros con mayor precisión, modelos espaciotemporales entrenados con videos de accidentes reales, una base de datos para trazabilidad histórica y un dashboard web más completo para monitoreo de múltiples cámaras. En una aplicación real también sería necesario incorporar anonimización, control de acceso y políticas claras de privacidad.

## Bibliografía

Shah, A. P., Lamare, J., Nguyen-Anh, T., & Hauptmann, A. (2018). *CADP: A Novel Dataset for CCTV Traffic Camera based Accident Analysis*. 15th IEEE International Conference on Advanced Video and Signal Based Surveillance.

Picek, L., Čermák, M., Hanzl, M., & Čermák, V. (2026). *ACCIDENT: A Benchmark Dataset for Vehicle Accident Detection from Traffic Surveillance Videos*. arXiv.

Yao, Y., Wang, X., Xu, M., Pu, Z., Atkins, E., & Crandall, D. (2023). *DoTA: Unsupervised Detection of Traffic Anomaly in Driving Videos*. IEEE Transactions on Pattern Analysis and Machine Intelligence.

Ultralytics. (2026). *Ultralytics YOLO Documentation*. https://docs.ultralytics.com/

OpenCV. (2026). *OpenCV Documentation*. https://docs.opencv.org/

PyTorch. (2026). *PyTorch Documentation*. https://pytorch.org/docs/

Streamlit. (2026). *Streamlit Documentation*. https://docs.streamlit.io/

## Anexos

### Anexo A. Código fuente utilizado

Los módulos principales del prototipo final se encuentran en:

| Archivo | Descripción |
|---|---|
| `app/main.py` | Interfaz Streamlit |
| `app/inference.py` | Inferencia por consola |
| `app/processing/pipeline.py` | Pipeline integrado de video |
| `app/detection/detector.py` | Detector YOLOv8 |
| `app/tracking/tracker.py` | Tracking persistente |
| `app/kinematics/calculator.py` | Cálculos cinemáticos |
| `app/collision/detector.py` | Detección de colisiones |
| `app/severity/classifier.py` | Clasificación de severidad |
| `app/visualization/visualizer.py` | Visualización y anotaciones |
| `config/settings.yaml` | Configuración del sistema |

### Anexo B. Datos utilizados

El repositorio incluye datos en las carpetas:

```text
datasets/
yolo_training/dataset/
```

También se incluye `generate_test_videos.py`, que crea videos sintéticos para pruebas de funcionamiento: flujo normal, prueba simple y simulación de colisión.

### Anexo C. Comandos de ejecución

Instalación recomendada:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Si el equipo no tiene GPU o la instalación intenta descargar paquetes CUDA demasiado grandes, se puede instalar el perfil CPU:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-cpu.txt
```

Ejecución con interfaz:

```bash
streamlit run app/main.py
```

Ejecución por consola:

```bash
python app/inference.py --video datasets/collision_test.mp4 --output outputs/evaluacion_final
```

### Anexo D. Archivos de salida

Por cada video procesado, el sistema puede generar:

```text
*_processed.mp4   Video anotado con detecciones y alertas
*_results.json    Resumen completo del procesamiento
*_events.csv      Tabla de accidentes detectados
*_summary.json    Resumen de inferencia por consola
```
