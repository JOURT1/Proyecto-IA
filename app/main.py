"""
Streamlit Web Interface for Traffic Accident Detection System
Main application entry point
"""

import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import time
from typing import Optional
import json

# Configure page
st.set_page_config(
    page_title="Sistema Inteligente de Detección de Accidentes",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .accident-severe {
        background-color: #ff6b6b;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        font-size: 18px;
        font-weight: bold;
    }
    .accident-moderate {
        background-color: #ffa500;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        font-size: 18px;
        font-weight: bold;
    }
    .accident-light {
        background-color: #ffff00;
        color: black;
        padding: 1rem;
        border-radius: 0.5rem;
        font-size: 18px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Import processing modules
try:
    from app.processing.pipeline import VideoProcessor
    from app.utils.config import Config, get_logger
    from app.utils.adaptive_learner import AdaptiveLearner
    from auto_retrain_monitor import AutoRetrainingManager
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.stop()

# Initialize components
@st.cache_resource
def get_processor():
    """Get or create VideoProcessor instance."""
    return VideoProcessor()

@st.cache_resource
def get_config():
    """Get or create Config instance."""
    config = Config()
    config.load_config("config/settings.yaml")
    return config

@st.cache_resource
def get_learner():
    """Get or create AdaptiveLearner instance."""
    return AdaptiveLearner("config/learning_history.json")

@st.cache_resource
def get_retrainer():
    """Get or create AutoRetrainingManager instance."""
    return AutoRetrainingManager()

# Header
st.title("🚗 Sistema Inteligente de Detección de Accidentes de Tránsito")
st.markdown("### Detección automática de colisiones vehiculares mediante Visión Artificial")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuración del Sistema")
    
    st.markdown("---")
    
    st.subheader("📊 Información del Sistema")
    st.info("""
    **Stack Tecnológico:**
    - Python 3.13
    - YOLO v8n (Detección)
    - ByteTrack (Tracking)
    - OpenCV (Procesamiento)
    - Streamlit (Interfaz)
    
    **Módulos:**
    - 🚗 Detección de vehículos
    - 📍 Tracking temporal
    - 📐 Cálculo cinemático
    - 💥 Detección de colisiones
    - 🔴 Clasificación de severidad
    """)

# ============ RE-ENTRENAMIENTO EN PROGRESO ============
if st.session_state.get('retraining_in_progress', False):
    st.info("🤖 Re-entrenamiento iniciado en background. Puedes continuar usando el sistema.")
    st.info("⏳ Esto puede tardar 10-30 minutos. Verifica el progreso en Tab 4 - Información.")
    st.session_state.retraining_in_progress = False  # Reset para evitar múltiples ejecuciones

# Main content
st.markdown("---")

# Tab sections
tab1, tab2, tab3, tab4 = st.tabs(
    ["📹 Carga de Video", "🔍 Análisis", "📊 Resultados", "ℹ️ Información"]
)

# ==================== TAB 1: VIDEO UPLOAD ====================
with tab1:
    st.header("Carga de Video de Vigilancia CCTV")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Selecciona un video CCTV (formatos: MP4, AVI, MOV, MKV)",
            type=["mp4", "avi", "mov", "mkv"],
            help="Máximo 500 MB. Video de cámara fija de vigilancia urbana."
        )
    
    with col2:
        st.info("**Requisitos:**\n- Cámara fija CCTV\n- Vigilancia urbana\n- Máx 500 MB")
    
    if uploaded_file is not None:
        st.success(f"✅ Archivo cargado: {uploaded_file.name} ({uploaded_file.size / (1024*1024):.1f} MB)")
        
        # Save uploaded file to project folder
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        temp_video_path = str(temp_dir / uploaded_file.name)
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Display video preview
        st.subheader("Vista previa del video")
        cap = cv2.VideoCapture(temp_video_path)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(frame_rgb, caption="Primer frame del video", width=640)
        
        # Get video info
        cap = cv2.VideoCapture(temp_video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        cap.release()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📐 Resolución", f"{width}×{height}")
        with col2:
            st.metric("⏱️ FPS", f"{fps:.1f}")
        with col3:
            st.metric("🎬 Frames", f"{total_frames:,}")
        with col4:
            st.metric("⏳ Duración", f"{int(duration//60)}:{int(duration%60):02d}")
        
        # Store in session state
        st.session_state.video_path = temp_video_path
        st.session_state.video_name = Path(uploaded_file.name).stem
        st.session_state.video_info = {
            'width': width,
            'height': height,
            'fps': fps,
            'total_frames': total_frames,
            'duration': duration
        }

# ==================== TAB 2: ANALYSIS ====================
with tab2:
    st.header("Procesamiento y Análisis")
    
    if 'video_path' not in st.session_state:
        st.warning("⚠️ Por favor, carga un video primero en la pestaña '📹 Carga de Video'")
        st.stop()
    
    # Obtener learner
    learner = get_learner()
    current_params = learner.get_current_params()
    stats = learner.get_stats()
    
    # Mostrar estado de aprendizaje
    st.info("🤖 **Sistema de Aprendizaje Automático Activo**")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📚 Análisis previos", stats['total_analyses'])
    with col2:
        st.metric("⭐ Calidad promedio", f"{stats['avg_quality']:.0%}")
    with col3:
        st.metric("🎯 Mejor resultado", f"{stats['best_quality']:.0%}")
    with col4:
        st.metric("🔄 Progreso", f"{stats['learning_progress']}%")
    
    st.markdown("---")
    
    # Mostrar parámetros auto-ajustados (SOLO LECTURA)
    st.subheader("🔧 Parámetros Auto-Ajustados")
    recommendation = learner.get_recommendations()
    st.write(f"📊 {recommendation}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "🎯 Confianza",
            f"{current_params['confidence_threshold']:.2f}",
            delta="Auto-optimizado",
            delta_color="off"
        )
    with col2:
        st.metric(
            "📍 IoU (NMS)",
            f"{current_params['iou_threshold']:.2f}",
            delta="Auto-optimizado",
            delta_color="off"
        )
    with col3:
        st.metric(
            "⚠️ Sensibilidad",
            f"{current_params['collision_sensitivity']:.2f}",
            delta="Auto-optimizado",
            delta_color="off"
        )
    
    st.markdown("---")
    
    # Guardar parámetros en sesión
    confidence_threshold = current_params['confidence_threshold']
    iou_threshold = current_params['iou_threshold']
    collision_sensitivity = current_params['collision_sensitivity']
    
    st.markdown("---")
    
    # Export options
    st.subheader("💾 Opciones de Exportación")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        export_video = st.checkbox("📹 Exportar video procesado", value=True)
    with col2:
        export_json = st.checkbox("📄 Exportar JSON", value=True)
    with col3:
        export_csv = st.checkbox("📊 Exportar CSV", value=True)
    st.markdown("---")
    
    # Process button
    st.subheader("▶️ Iniciar Procesamiento")
    
    if st.button("🚀 Iniciar Análisis", use_container_width=True, type="primary"):
        # Create layout for real-time display
        col_left, col_right = st.columns([2, 1])
        
        with col_left:
            st.write("🔄 **Procesando video...**")
            progress_bar = st.progress(0)
            progress_text = st.empty()
            frame_container = st.empty()
        
        with col_right:
            st.write("📊 **Estadísticas en vivo**")
            metrics_col1, metrics_col2 = st.columns(2)
            with metrics_col1:
                frame_counter = st.empty()
                accidents_counter = st.empty()
            with metrics_col2:
                fps_counter = st.empty()
                progress_percent_text = st.empty()
        
        try:
            video_path = st.session_state.video_path
            
            # Validate video
            if not Path(video_path).exists():
                st.error("❌ El archivo de video no se encontró")
                st.stop()
            
            # Get processor
            processor = get_processor()
            processor.detector.set_confidence_threshold(confidence_threshold)
            processor.detector.set_iou_threshold(iou_threshold)
            
            # Adjust collision detection sensitivity
            # Higher sensitivity = lower detection threshold = fewer false positives
            critical_distance = 1.5 * collision_sensitivity  # 1.05 to 1.5 meters
            processor.collision_detector.config.set('collision.critical_proximity_distance', critical_distance)
            
            # Prepare output
            output_dir = Path("outputs")
            output_dir.mkdir(exist_ok=True)
            
            output_video = None
            if export_video:
                output_video = str(output_dir / f"{st.session_state.video_name}_processed.mp4")
            
            # Callback for progress updates
            def on_progress(progress, current_frame, total_frames, accidents, fps):
                progress_bar.progress(min(progress / 100.0, 0.99))
                progress_text.write(
                    f"📊 **Progreso:** {int(progress)}% | "
                    f"Frame {current_frame:,}/{total_frames:,}"
                )
                frame_counter.write(f"**📹 Frame:** {current_frame:,}")
                accidents_counter.write(f"**⚠️ Accidentes:** {accidents}")
                fps_counter.write(f"**⏱️ FPS:** {fps:.1f}")
                progress_percent_text.write(f"**%:** {int(progress)}%")
            
            # Callback to show current frame
            def on_frame(frame, frame_idx, accidents_in_frame):
                # Convert BGR to RGB for display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize for display
                h, w = frame_rgb.shape[:2]
                if w > 640:
                    scale = 640 / w
                    frame_rgb = cv2.resize(frame_rgb, (640, int(h * scale)))
                
                frame_container.image(
                    frame_rgb, 
                    caption=f"🎬 Frame {frame_idx:,} ({accidents_in_frame} colisiones detectadas)",
                    use_column_width=True
                )
            
            # Process video with callbacks
            with st.spinner("⏳ Procesando (esto puede tomar varios minutos)..."):
                results = processor.process_video(
                    video_path,
                    output_video_path=output_video,
                    visualize=True,
                    export_json=export_json,
                    export_csv=export_csv,
                    progress_callback=on_progress,
                    frame_callback=on_frame
                )
            
            # Store results
            st.session_state.results = results
            
            # Finalize progress
            progress_bar.progress(1.0)
            progress_text.write(
                f"✅ **Completado:** {results['total_frames']} frames | "
                f"⏱️ {results['total_time_seconds']:.1f}s | "
                f"⚠️ {results['accidents_detected']} accidentes"
            )
            
            # Success section
            st.markdown("---")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📊 Frames", f"{results['total_frames']:,}")
            with col2:
                st.metric("⏱️ Tiempo", f"{results['total_time_seconds']:.1f}s")
            with col3:
                st.metric("⚠️ Accidentes", results['accidents_detected'])
            with col4:
                st.metric("🎬 FPS", f"{results['fps']:.1f}")
            
            st.success(f"""
            ✅ **Análisis completado exitosamente**
            - 🚗 Vehículos detectados: {results['vehicles_tracked']}
            - ⚠️ Accidentes detectados: **{results['accidents_detected']}**
            - 📁 Archivos guardados en `outputs/`
            """)
            
            # Sistema de feedback para aprendizaje
            st.markdown("---")
            st.subheader("🧠 Feedback para Aprendizaje")
            st.write("¿Qué tan precisos fueron los resultados?")
            
            feedback_col1, feedback_col2 = st.columns(2)
            with feedback_col1:
                quality_rating = st.slider(
                    "Calidad del análisis",
                    min_value=0,
                    max_value=100,
                    value=75,
                    step=5,
                    help="0%=pobre (muchos falsos), 100%=perfecto",
                    key="quality_feedback"
                )
            
            with feedback_col2:
                if st.button("✅ Guardar Feedback", type="primary"):
                    learner.record_analysis(
                        video_name=st.session_state.video_name,
                        accidents_detected=results['accidents_detected'],
                        total_frames=results['total_frames'],
                        confidence_threshold=confidence_threshold,
                        iou_threshold=iou_threshold,
                        collision_sensitivity=collision_sensitivity,
                        false_positive_feedback=quality_rating / 100.0
                    )
                    st.success("🤖 Feedback guardado. El sistema mejorará en próximos análisis.")
                    
                    # Chequear si necesita re-entrenamiento automático
                    retrainer = get_retrainer()
                    if retrainer.check_need_retraining(threshold_quality=0.65, min_analyses=3):
                        st.warning("⚠️ Calidad baja detectada - Se recomienda re-entrenamiento")
                        st.info("Ve a Tab 4 (Información) para iniciar re-entrenamiento automático")
                    
                    st.balloons()
            
            st.session_state.show_results = True
        
        except Exception as e:
            st.error(f"❌ Error durante el procesamiento: {str(e)}")
            import traceback
            with st.expander("📋 Detalles del error"):
                st.code(traceback.format_exc())

# ==================== TAB 3: RESULTS ====================
with tab3:
    st.header("Resultados del Análisis")
    
    if 'results' not in st.session_state:
        st.info("Procesa un video primero para ver los resultados")
    else:
        results = st.session_state.results
        
        # Summary metrics
        st.subheader("📈 Resumen Ejecutivo")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Frames Procesados", f"{results['total_frames']}")
        with col2:
            st.metric("Tiempo Total", f"{results['total_time_seconds']:.1f}s")
        with col3:
            st.metric("FPS Procesamiento", f"{results['fps']:.1f}")
        with col4:
            st.metric("Vehículos Detectados", f"{results['vehicles_tracked']}")
        with col5:
            st.metric("🚨 Accidentes", f"{results['accidents_detected']}", 
                     delta=f"{results['accidents_detected']} evento(s)")
        
        # Detailed metrics
        if results['metrics']:
            st.subheader("📊 Métricas de Desempeño")
            metrics = results['metrics']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Tiempo Promedio Detección", 
                         f"{metrics.get('avg_detection_time_ms', 0):.2f} ms")
            with col2:
                st.metric("Tiempo Promedio Tracking",
                         f"{metrics.get('avg_tracking_time_ms', 0):.2f} ms")
            with col3:
                st.metric("Tiempo Promedio Colisión",
                         f"{metrics.get('avg_collision_detection_time_ms', 0):.2f} ms")
        
        # Detected accidents list
        if results['detected_accidents']:
            st.subheader("🚨 Accidentes Detectados")
            
            for idx, accident in enumerate(results['detected_accidents'], 1):
                severity = accident['severity']
                
                # Color based on severity
                if severity == "Severo":
                    st.markdown(f'<div class="accident-severe">Evento {idx}: ACCIDENTE SEVERO</div>', 
                               unsafe_allow_html=True)
                elif severity == "Moderado":
                    st.markdown(f'<div class="accident-moderate">Evento {idx}: ACCIDENTE MODERADO</div>',
                               unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="accident-light">Evento {idx}: INCIDENTE LEVE</div>',
                               unsafe_allow_html=True)
                
                # Details
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.write(f"⏱️ **Timestamp:** {accident['timestamp']}")
                with col2:
                    st.write(f"🔢 **Frame:** {accident['frame']}")
                with col3:
                    st.write(f"🏎️ **Vehículos:** {accident['vehicle_ids']}")
                with col4:
                    st.write(f"📊 **Score:** {accident['score']:.1f}/100")
                
                st.write(f"**Descripción:** {accident['description']}")
                st.divider()
        else:
            st.success("✅ No se detectaron accidentes en el video")
        
        # Download results
        st.subheader("⬇️ Descargar Resultados")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            output_dir = Path("outputs")
            video_file = output_dir / f"{st.session_state.get('video_name', 'video')}_processed.mp4"
            if video_file.exists():
                with open(video_file, "rb") as f:
                    st.download_button(
                        "📹 Descargar Video Procesado",
                        f.read(),
                        file_name=video_file.name,
                        mime="video/mp4"
                    )
            else:
                st.info("Video procesado no disponible")
        
        with col2:
            json_file = output_dir / f"{st.session_state.get('video_name', 'video')}_results.json"
            if json_file.exists():
                with open(json_file, "rb") as f:
                    st.download_button(
                        "📄 Descargar JSON",
                        f.read(),
                        file_name=json_file.name,
                        mime="application/json"
                    )
        
        with col3:
            csv_file = output_dir / f"{st.session_state.get('video_name', 'video')}_events.csv"
            if csv_file.exists():
                with open(csv_file, "rb") as f:
                    st.download_button(
                        "📊 Descargar CSV",
                        f.read(),
                        file_name=csv_file.name,
                        mime="text/csv"
                    )

# ==================== TAB 4: INFO ====================
with tab4:
    st.header("ℹ️ Información del Sistema")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Características")
        st.markdown("""
        - ✅ Detección de vehículos en tiempo real
        - ✅ Tracking persistente con ByteTrack
        - ✅ Cálculo de trayectorias y cinemática
        - ✅ Detección de colisiones basada en reglas
        - ✅ Clasificación automática de severidad
        - ✅ Visualización en tiempo real
        - ✅ Exportación de resultados (JSON, CSV, Video)
        - ✅ Interfaz web intuitiva
        """)
        
        st.subheader("🔧 Stack Tecnológico")
        st.markdown("""
        - **Lenguaje:** Python 3.11+
        - **Detección:** Ultralytics YOLOv8
        - **Tracking:** ByteTrack
        - **Procesamiento:** OpenCV
        - **Deep Learning:** PyTorch
        - **Interfaz:** Streamlit
        - **Configuración:** YAML
        """)
    
    with col2:
        st.subheader("🎯 Casos de Uso")
        st.markdown("""
        - Monitoreo de intersecciones urbanas
        - Vigilancia de vías de tráfico
        - Análisis automático de accidentes
        - Generación de alertas en tiempo real
        - Investigación de incidentes
        - Mejora de seguridad vial
        """)
        
        st.subheader("⚠️ Limitaciones")
        st.markdown("""
        - Estimación de velocidad aproximada (sin calibración real)
        - Severidad estimada, no física real
        - Puede fallar con iluminación extrema
        - Errores por oclusión severa
        - Prototipo académico (no certificado)
        - Requiere cámara fija CCTV
        """)
    
    st.divider()
    
    # ============ MODELO Y RE-ENTRENAMIENTO ============
    st.subheader("🤖 Estado del Modelo - Auto-Entrenamiento")
    
    retrainer = get_retrainer()
    learner = get_learner()
    
    col_model1, col_model2 = st.columns(2)
    
    with col_model1:
        st.markdown("**📊 Estadísticas de Aprendizaje**")
        stats = learner.get_stats()
        
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("📚 Análisis", stats['total_analyses'])
        with m2:
            st.metric("⭐ Calidad", f"{stats['avg_quality']:.0%}")
        with m3:
            st.metric("🎯 Mejor", f"{stats['best_quality']:.0%}")
        with m4:
            st.metric("🔄 Progreso", f"{stats['learning_progress']:.0f}%")
    
    with col_model2:
        st.markdown("**🏋️ Estadísticas de Re-Entrenamiento**")
        retrain_stats = retrainer.get_retraining_stats()
        
        r1, r2, r3, r4 = st.columns(4)
        with r1:
            st.metric("🔁 Re-trainings", retrain_stats['total_retrainings'])
        with r2:
            st.metric("✅ Exitosos", retrain_stats['successful'])
        with r3:
            st.metric("❌ Fallidos", retrain_stats['failed'])
        with r4:
            st.metric("⏱️ Último", "Nunca" if not retrain_stats['last_retraining'] else "Hoy")
    
    # Sistema de auto-detección de necesidad de re-entrenamiento
    st.markdown("**🚨 Verificación Automática de Necesidad de Re-Entrenamiento**")
    
    col_check1, col_check2 = st.columns([3, 1])
    
    with col_check1:
        needs_retrain = retrainer.check_need_retraining(threshold_quality=0.60)
        
        if needs_retrain:
            st.warning("""
            ⚠️ **Calidad detectada por debajo del umbral**
            
            El sistema ha detectado muchos falsos positivos o calidad baja en análisis recientes.
            Se recomienda re-entrenar el modelo para mejorar precisión.
            """)
        else:
            st.success("""
            ✅ **Calidad de detección es buena**
            
            El modelo está funcionando correctamente. No se necesita re-entrenamiento en este momento.
            """)
    
    with col_check2:
        if st.button("🔄 Verificar Ahora", use_container_width=True):
            with st.spinner("Verificando..."):
                needs_retrain = retrainer.check_need_retraining(threshold_quality=0.60)
            st.rerun()
    
    st.markdown("---")
    
    # Botón de re-entrenamiento manual
    st.markdown("**🚀 Re-Entrenamiento Manual**")
    st.markdown("""
    Inicia un ciclo de re-entrenamiento para mejorar la precisión del modelo.
    El proceso puede tardar 10-30 minutos dependiendo del hardware.
    """)
    
    col_train1, col_train2, col_train3 = st.columns([1, 1, 2])
    
    with col_train1:
        if st.button("🏋️ Iniciar Re-Entrenamiento", use_container_width=True, type="primary"):
            st.session_state.retraining_in_progress = True
    
    with col_train2:
        if st.button("📋 Ver Historial", use_container_width=True):
            st.session_state.show_retrain_history = True
    
    # Mostrar historial si se solicita
    if st.session_state.get('show_retrain_history', False):
        st.markdown("**📋 Historial de Re-Entrenamientos**")
        retrain_log = Path("config/retraining_log.json")
        
        if retrain_log.exists():
            try:
                with open(retrain_log) as f:
                    retrain_history = json.load(f)
                
                if retrain_history:
                    for i, event in enumerate(reversed(retrain_history[-10:])):  # Últimos 10
                        status_emoji = "✅" if event['status'] == 'success' else "❌"
                        st.write(f"{status_emoji} {event['timestamp']} - {event['status'].upper()}")
                else:
                    st.info("Sin eventos de re-entrenamiento registrados")
            except:
                st.warning("No se pudo cargar el historial")
        else:
            st.info("Sin eventos de re-entrenamiento registrados")
    
    st.divider()
    
    st.subheader("📖 Guía de Uso")
    st.markdown("""
    1. **Carga de Video:** Sube un archivo de vigilancia CCTV
    2. **Configuración:** Ajusta los umbrales de detección (opcional)
    3. **Análisis:** Inicia el procesamiento del video
    4. **Revisión:** Examina los resultados y accidentes detectados
    5. **Descarga:** Obtén el video procesado y resultados
    
    ### Severidad de Accidentes
    
    - **🟢 Leve:** Rozón, contacto menor, impacto bajo
    - **🟡 Moderado:** Colisión perceptible, cambio parcial de movimiento
    - **🔴 Severo:** Choque de alto impacto, cambio fuerte de trayectoria
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <small>
    Sistema Inteligente de Detección de Accidentes de Tránsito | 
    Asignatura: Inteligencia Artificial 1 | 
    UDLA - Ingeniería de Software
    </small>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'processing' not in st.session_state:
    st.session_state.processing = False
