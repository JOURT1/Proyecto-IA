"""
Generador de videos de prueba para el sistema de deteccin de accidentes
Crea 3 videos de ejemplo para probar el sistema
"""

import cv2
import numpy as np
import os
from pathlib import Path

def create_test_video(filename, duration_seconds=10, fps=30, width=640, height=480):
    """Crear un video de prueba con contenido dinmico"""
    
    output_path = f"datasets/{filename}"
    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )
    
    total_frames = duration_seconds * fps
    
    print(f"Generando: {filename} ({duration_seconds}s, {total_frames} frames)")
    
    for frame_num in range(total_frames):
        # Crear frame con fondo gris
        frame = np.full((height, width, 3), 100, dtype=np.uint8)
        
        # Agregar informacin del frame
        cv2.putText(
            frame, 
            f"Test Video - Frame {frame_num}",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        # Dibujar rectngulos mviles para simular vehculos
        x = int(50 + (frame_num % 100) * 5)
        y = 150
        cv2.rectangle(frame, (x, y), (x+80, y+60), (0, 0, 255), 2)
        cv2.putText(frame, "CAR1", (x+10, y+35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Segundo vehculo
        x2 = int(400 - (frame_num % 100) * 3)
        y2 = 250
        cv2.rectangle(frame, (x2, y2), (x2+70, y2+50), (255, 0, 0), 2)
        cv2.putText(frame, "CAR2", (x2+5, y2+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Informacin del tiempo
        cv2.putText(
            frame,
            f"Time: {frame_num/fps:.2f}s",
            (width-200, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            1
        )
        
        # FPS
        cv2.putText(
            frame,
            f"FPS: {fps}",
            (width-200, height-50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            1
        )
        
        out.write(frame)
        
        # Mostrar progreso
        if (frame_num + 1) % 30 == 0:
            print(f"  - {frame_num + 1}/{total_frames} frames generados")
    
    out.release()
    print(f" Guardado: {output_path}\n")
    return output_path


def create_collision_simulation_video(filename="collision_test.mp4"):
    """Crear video con simulacin de colisin"""
    
    output_path = f"datasets/{filename}"
    fps = 30
    width, height = 640, 480
    duration = 15  # 15 segundos
    
    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )
    
    total_frames = duration * fps
    print(f"Generando: {filename} (Simulacion de colision, {duration}s)")
    
    for frame_num in range(total_frames):
        frame = np.full((height, width, 3), 80, dtype=np.uint8)
        
        # Carril de trnsito
        cv2.line(frame, (0, 240), (width, 240), (200, 200, 200), 2)
        
        # Ttulo
        cv2.putText(frame, "COLLISION SIMULATION", (50, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Vehculo 1: movindose de izquierda a derecha
        x1 = int(50 + frame_num * 2)
        y1 = 220
        if x1 < width:
            cv2.rectangle(frame, (x1, y1), (x1+80, y1+60), (0, 255, 0), 2)
            cv2.putText(frame, "V1", (x1+20, y1+35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        # Vehculo 2: movindose de derecha a izquierda (hacia colisin)
        x2 = int(width - 50 - frame_num * 1.5)
        y2 = 220
        if x2 > 0:
            cv2.rectangle(frame, (x2, y2), (x2+80, y2+60), (0, 0, 255), 2)
            cv2.putText(frame, "V2", (x2+20, y2+35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        # Detectar colisin (cuando se superponen)
        if x1 + 80 > x2 and x2 + 80 > x1:
            # Colisin detectada
            collision_y = int(240 + 30 * np.sin(frame_num * 0.3))
            cv2.circle(frame, (int((x1 + x2) / 2), collision_y), 30, (0, 0, 255), -1)
            cv2.putText(frame, "COLLISION!", (int((x1 + x2) / 2) - 50, collision_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Informacin
        cv2.putText(frame, f"Frame: {frame_num} | Time: {frame_num/fps:.2f}s",
                   (50, height-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        out.write(frame)
        
        if (frame_num + 1) % 30 == 0:
            print(f"  - {frame_num + 1}/{total_frames} frames generados")
    
    out.release()
    print(f" Guardado: {output_path}\n")
    return output_path


def create_traffic_flow_video(filename="traffic_flow.mp4"):
    """Crear video de flujo de trnsito normal"""
    
    output_path = f"datasets/{filename}"
    fps = 30
    width, height = 640, 480
    duration = 12
    
    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )
    
    total_frames = duration * fps
    print(f"Generando: {filename} (Flujo de transito, {duration}s)")
    
    # Definir posiciones de carriles
    lanes = [150, 250, 350]
    
    for frame_num in range(total_frames):
        frame = np.full((height, width, 3), 60, dtype=np.uint8)
        
        # Carreteras (lneas divisoras)
        for lane in lanes:
            cv2.line(frame, (0, lane), (width, lane), (200, 200, 0), 2)
        
        # Ttulo
        cv2.putText(frame, "TRAFFIC FLOW", (50, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Vehculos en cada carril
        colors = [(0, 255, 0), (0, 255, 255), (255, 0, 0)]
        labels = ["CAR1", "CAR2", "CAR3"]
        
        for lane_idx, lane_y in enumerate(lanes):
            # Mltiples vehculos en el mismo carril
            for car_idx in range(3):
                x = int(50 + (frame_num * (2 + car_idx * 0.5)) % width)
                y = lane_y - 30
                
                if x > -100 and x < width:
                    cv2.rectangle(frame, (x, y), (x+60, y+50), colors[lane_idx], 2)
                    cv2.putText(frame, f"C{car_idx+1}", (x+10, y+30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Info
        cv2.putText(frame, f"Frame: {frame_num} | Time: {frame_num/fps:.2f}s",
                   (50, height-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        out.write(frame)
        
        if (frame_num + 1) % 30 == 0:
            print(f"  - {frame_num + 1}/{total_frames} frames generados")
    
    out.release()
    print(f" Guardado: {output_path}\n")
    return output_path


if __name__ == "__main__":
    print("=" * 60)
    print("GENERADOR DE VIDEOS DE PRUEBA")
    print("=" * 60)
    print()
    
    # Crear carpeta si no existe
    os.makedirs("datasets", exist_ok=True)
    
    # Generar 3 videos de prueba
    videos = []
    
    # Video 1: Simple
    videos.append(create_test_video("test_simple.mp4", duration_seconds=10))
    
    # Video 2: Simulacion de colision
    videos.append(create_collision_simulation_video("collision_test.mp4"))
    
    # Video 3: Flujo de transito
    videos.append(create_traffic_flow_video("traffic_flow.mp4"))
    
    print("=" * 60)
    print("VIDEOS CREADOS EXITOSAMENTE")
    print("=" * 60)
    print("\nUbicacion: c:\\Users\\USER\\Desktop\\Proyecto IA\\datasets\\")
    print("\nVideos generados:")
    for i, video in enumerate(videos, 1):
        size_mb = os.path.getsize(video) / (1024*1024)
        print(f"   {i}. {os.path.basename(video)} ({size_mb:.1f} MB)")
    
    print("\nProximos pasos:")
    print("   1. Abre http://localhost:8501")
    print("   2. En la pestana 'Carga de Video'")
    print("   3. Sube uno de estos videos")
    print("   4. Haz clic en 'Iniciar Analisis'")
    print("   5. Observa los resultados!")
    print()
