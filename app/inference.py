"""
Inference Script for Accident Detection System
Run detection on a video file and generate results
"""

import argparse
import sys
from pathlib import Path
import json

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.processing.pipeline import VideoProcessor
from app.utils.config import Config, get_logger

def main():
    """Main inference function."""
    
    parser = argparse.ArgumentParser(
        description="Run accident detection inference on a video"
    )
    
    parser.add_argument(
        '--video',
        required=True,
        help='Path to input video file'
    )
    
    parser.add_argument(
        '--output',
        default='outputs/',
        help='Output directory for results (default: outputs/)'
    )
    
    parser.add_argument(
        '--config',
        default='config/settings.yaml',
        help='Path to configuration file (default: config/settings.yaml)'
    )
    
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.5,
        help='Detection confidence threshold (default: 0.5)'
    )
    
    parser.add_argument(
        '--iou',
        type=float,
        default=0.45,
        help='IoU threshold for NMS (default: 0.45)'
    )
    
    parser.add_argument(
        '--export-video',
        action='store_true',
        default=True,
        help='Export annotated video'
    )
    
    parser.add_argument(
        '--export-json',
        action='store_true',
        default=True,
        help='Export results as JSON'
    )
    
    parser.add_argument(
        '--export-csv',
        action='store_true',
        default=True,
        help='Export events as CSV'
    )
    
    args = parser.parse_args()
    
    # Setup logger
    logger = get_logger()
    
    logger.info("=" * 80)
    logger.info("TRAFFIC ACCIDENT DETECTION SYSTEM - INFERENCE")
    logger.info("=" * 80)
    
    # Check if video exists
    video_path = Path(args.video)
    if not video_path.exists():
        logger.error(f"Video file not found: {args.video}")
        sys.exit(1)
    
    logger.info(f"Input video: {args.video}")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Output directory: {args.output}")
    
    # Initialize processor
    try:
        processor = VideoProcessor(args.config)
        logger.info("VideoProcessor initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize processor: {e}")
        sys.exit(1)
    
    # Update thresholds
    processor.detector.set_confidence_threshold(args.confidence)
    processor.detector.set_iou_threshold(args.iou)
    logger.info(f"Detection thresholds set: confidence={args.confidence}, iou={args.iou}")
    
    # Prepare output path
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    video_stem = video_path.stem
    output_video = str(output_path / f"{video_stem}_processed.mp4") if args.export_video else None
    
    # Process video
    try:
        logger.info("Starting video processing...")
        results = processor.process_video(
            str(video_path),
            output_video_path=output_video,
            visualize=True,
            export_json=args.export_json,
            export_csv=args.export_csv
        )
        
        logger.info("=" * 80)
        logger.info("PROCESSING COMPLETED")
        logger.info("=" * 80)
        
        # Print summary
        logger.info(f"Total frames processed: {results['total_frames']}")
        logger.info(f"Processing time: {results['total_time_seconds']:.2f} seconds")
        logger.info(f"Average FPS: {results['fps']:.2f}")
        logger.info(f"Vehicles tracked: {results['vehicles_tracked']}")
        logger.info(f"Accidents detected: {results['accidents_detected']}")
        
        if results['detected_accidents']:
            logger.info("\nDetected accidents:")
            for i, accident in enumerate(results['detected_accidents'], 1):
                logger.info(f"  {i}. {accident['timestamp']} - {accident['severity']}")
                logger.info(f"     Vehicles: {accident['vehicle_ids']}")
                logger.info(f"     Score: {accident['score']:.1f}/100")
        
        # Save summary
        summary_path = output_path / f"{video_stem}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nSummary saved to: {summary_path}")
        
        if output_video:
            logger.info(f"Processed video saved to: {output_video}")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error during processing: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
