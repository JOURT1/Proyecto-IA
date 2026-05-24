"""
Descargador Automático de Dataset CCTV desde Kaggle
Descarga videos CCTV reales y los organiza en la carpeta datasets
"""

import kagglehub
import os
from pathlib import Path
import shutil

def download_cctv_dataset():
    """Descargar dataset CCTV de Kaggle"""
    print("=" * 70)
    print("📥 DESCARGANDO DATASET CCTV DE KAGGLE")
    print("=" * 70)
    print()
    
    try:
        print("🔄 Descargando: fahaddalwai/cctvfootagevideo")
        print("   Esto puede tomar unos minutos según tu conexión...")
        print()
        
        # Descargar dataset
        path = kagglehub.dataset_download("fahaddalwai/cctvfootagevideo")
        
        print(f"\n✅ Dataset descargado en: {path}")
        
        # Listar archivos
        print("\n📁 Archivos encontrados:")
        for root, dirs, files in os.walk(path):
            level = root.replace(path, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f'{indent}{os.path.basename(root)}/')
            subindent = ' ' * 2 * (level + 1)
            for file in files[:10]:  # Mostrar máximo 10 archivos
                print(f'{subindent}{file}')
            if len(files) > 10:
                print(f'{subindent}... y {len(files) - 10} archivos más')
        
        # Copiar videos a datasets
        print("\n" + "=" * 70)
        print("📂 COPIANDO VIDEOS A CARPETA DE PROYECTO")
        print("=" * 70)
        
        datasets_dir = Path.cwd() / "datasets"
        datasets_dir.mkdir(exist_ok=True)
        
        # Buscar archivos de video
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv']
        video_count = 0
        
        for root, dirs, files in os.walk(path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in video_extensions):
                    src = os.path.join(root, file)
                    dst = datasets_dir / file
                    
                    print(f"\n📹 Copiando: {file}")
                    try:
                        # Mostrar tamaño
                        size_mb = os.path.getsize(src) / (1024*1024)
                        print(f"   Tamaño: {size_mb:.1f} MB")
                        
                        # Copiar
                        shutil.copy2(src, dst)
                        print(f"   ✓ Guardado en: datasets/{file}")
                        video_count += 1
                    except Exception as e:
                        print(f"   ❌ Error: {e}")
        
        print("\n" + "=" * 70)
        print(f"✅ DESCARGA COMPLETADA")
        print("=" * 70)
        print(f"\n📊 Resumen:")
        print(f"   📹 Videos copiados: {video_count}")
        print(f"   📁 Ubicación: {datasets_dir}")
        print(f"\n🎯 Próximos pasos:")
        print(f"   1. Abre http://localhost:8501")
        print(f"   2. En '📹 Carga de Video'")
        print(f"   3. Selecciona uno de los videos descargados")
        print(f"   4. Haz clic en 'Iniciar Análisis'")
        print()
        
        return video_count > 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nPosibles causas:")
        print("  - No tienes credenciales de Kaggle configuradas")
        print("  - La conexión a internet es lenta")
        print("  - El dataset no existe")
        print("\nSolución:")
        print("  1. Ve a: https://www.kaggle.com/account/api")
        print("  2. Haz clic en 'Create New API Token'")
        print("  3. Guarda kaggle.json en: %USERPROFILE%\\.kaggle\\")
        return False


if __name__ == "__main__":
    download_cctv_dataset()
