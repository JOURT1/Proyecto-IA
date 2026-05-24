"""
Example Usage Script
Demonstrates how to use the Traffic Accident Detection System
"""

from app.processing.pipeline import VideoProcessor
from app.utils.config import Config, get_logger
import json


def example_1_basic_inference():
    """Example 1: Basic video processing with default settings."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Video Inference")
    print("="*80)
    
    # Create processor with default configuration
    processor = VideoProcessor()
    
    # Process video
    results = processor.process_video(
        video_path="path/to/your/video.mp4",
        output_video_path="outputs/processed_video.mp4",
        visualize=True,
        export_json=True,
        export_csv=True
    )
    
    # Print summary
    print(f"\nProcessing completed!")
    print(f"Total frames: {results['total_frames']}")
    print(f"Processing time: {results['total_time_seconds']:.2f} seconds")
    print(f"Average FPS: {results['fps']:.2f}")
    print(f"Accidents detected: {results['accidents_detected']}")
    
    # Print detected accidents
    if results['detected_accidents']:
        print("\nDetected Accidents:")
        for i, accident in enumerate(results['detected_accidents'], 1):
            print(f"  {i}. Frame {accident['frame']} ({accident['timestamp']})")
            print(f"     Severity: {accident['severity']}")
            print(f"     Score: {accident['score']:.1f}/100")
            print(f"     Vehicles: {accident['vehicle_ids']}")


def example_2_custom_config():
    """Example 2: Processing with custom configuration."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Custom Configuration")
    print("="*80)
    
    # Load and modify configuration
    config = Config()
    config.load_config("config/settings.yaml")
    
    # Customize parameters
    config.set("model.confidence_threshold", 0.6)  # Stricter detection
    config.set("collision.min_signals_for_collision", 4)  # More strict
    config.set("severity.thresholds.severo_min", 70)  # Higher bar for severe
    
    # Create processor with modified config
    processor = VideoProcessor("config/settings.yaml")
    
    # Process video
    results = processor.process_video(
        "path/to/video.mp4",
        output_video_path="outputs/strict_detection.mp4"
    )
    
    print(f"Accidents with stricter parameters: {results['accidents_detected']}")


def example_3_frame_by_frame():
    """Example 3: Process single frames (real-time scenario)."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Frame-by-Frame Processing")
    print("="*80)
    
    import cv2
    
    processor = VideoProcessor()
    
    # Open video
    cap = cv2.VideoCapture("path/to/video.mp4")
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        result = processor.process_frame(frame)
        
        # Access results
        detections = result['detections']
        tracks = result['tracks']
        collisions = result['collisions']
        severity = result['severity']
        
        # Process data (e.g., send alerts, update UI)
        if collisions:
            for collision in collisions:
                if collision.track_ids in severity:
                    assessment = severity[collision.track_ids]
                    print(f"Frame {frame_idx}: {assessment.level.value} accident detected")
        
        frame_idx += 1
    
    cap.release()


def example_4_detailed_analysis():
    """Example 4: Detailed analysis of results."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Detailed Results Analysis")
    print("="*80)
    
    processor = VideoProcessor()
    results = processor.process_video("path/to/video.mp4", export_json=True)
    
    # Performance metrics
    metrics = results['metrics']
    print("\nPerformance Metrics:")
    print(f"  Total frames: {metrics['total_frames']}")
    print(f"  FPS: {metrics['fps']:.2f}")
    print(f"  Avg detection time: {metrics['avg_detection_time_ms']:.2f} ms")
    print(f"  Avg tracking time: {metrics['avg_tracking_time_ms']:.2f} ms")
    print(f"  Avg collision detection time: {metrics['avg_collision_detection_time_ms']:.2f} ms")
    
    # Accident statistics
    accidents = results['detected_accidents']
    print(f"\nAccident Statistics:")
    print(f"  Total accidents: {len(accidents)}")
    
    if accidents:
        # Group by severity
        severity_counts = {}
        for acc in accidents:
            sev = acc['severity']
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
        
        print(f"  By severity:")
        for sev, count in severity_counts.items():
            print(f"    - {sev}: {count}")
        
        # Average score
        avg_score = sum(a['score'] for a in accidents) / len(accidents)
        print(f"  Average score: {avg_score:.1f}/100")


def example_5_batch_processing():
    """Example 5: Process multiple videos."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Batch Processing")
    print("="*80)
    
    from pathlib import Path
    
    processor = VideoProcessor()
    
    # Find all MP4 files in a directory
    video_dir = Path("datasets/test_videos")
    video_files = list(video_dir.glob("*.mp4"))
    
    print(f"Found {len(video_files)} videos to process")
    
    # Process each video
    all_results = []
    for video_path in video_files:
        print(f"\nProcessing: {video_path.name}")
        
        results = processor.process_video(
            str(video_path),
            output_video_path=f"outputs/{video_path.stem}_processed.mp4",
            export_json=True
        )
        
        all_results.append({
            'video': video_path.name,
            'accidents': results['accidents_detected'],
            'fps': results['fps'],
            'duration': results['total_time_seconds']
        })
    
    # Summary
    print("\n" + "="*80)
    print("BATCH PROCESSING SUMMARY")
    print("="*80)
    total_accidents = sum(r['accidents'] for r in all_results)
    print(f"Total videos processed: {len(all_results)}")
    print(f"Total accidents detected: {total_accidents}")
    
    for result in all_results:
        print(f"  - {result['video']}: {result['accidents']} accidents")


def example_6_configuration_presets():
    """Example 6: Using configuration presets for different scenarios."""
    print("\n" + "="*80)
    print("EXAMPLE 6: Configuration Presets")
    print("="*80)
    
    config = Config()
    
    # Preset 1: High Sensitivity (catch all accidents)
    print("\nPreset 1: High Sensitivity")
    config.load_config("config/settings.yaml")
    config.set("model.confidence_threshold", 0.4)
    config.set("collision.min_signals_for_collision", 2)
    config.set("severity.thresholds.severo_min", 50)
    
    processor = VideoProcessor()
    results = processor.process_video("path/to/video.mp4")
    print(f"Detected: {results['accidents_detected']} accidents (sensitive)")
    
    # Preset 2: High Precision (only clear accidents)
    print("\nPreset 2: High Precision")
    config.load_config("config/settings.yaml")
    config.set("model.confidence_threshold", 0.7)
    config.set("collision.min_signals_for_collision", 5)
    config.set("severity.thresholds.severo_min", 80)
    
    processor = VideoProcessor()
    results = processor.process_video("path/to/video.mp4")
    print(f"Detected: {results['accidents_detected']} accidents (precise)")
    
    # Preset 3: Balanced (default)
    print("\nPreset 3: Balanced")
    config.load_config("config/settings.yaml")
    processor = VideoProcessor()
    results = processor.process_video("path/to/video.mp4")
    print(f"Detected: {results['accidents_detected']} accidents (balanced)")


# ============================================================================
# MAIN - Choose which example to run
# ============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*80)
    print("TRAFFIC ACCIDENT DETECTION SYSTEM - USAGE EXAMPLES")
    print("="*80)
    
    # Uncomment the example you want to run:
    
    # example_1_basic_inference()
    # example_2_custom_config()
    # example_3_frame_by_frame()
    # example_4_detailed_analysis()
    # example_5_batch_processing()
    # example_6_configuration_presets()
    
    print("\n" + "="*80)
    print("Note: Uncomment the example you want to run in the main block")
    print("="*80)
