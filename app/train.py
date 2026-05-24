"""
Training Script for YOLO Fine-tuning on Traffic Accident Detection
"""

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO

def main():
    """Main training function."""
    
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLO model for vehicle detection"
    )
    
    parser.add_argument(
        '--data',
        default='config/dataset.yaml',
        help='Path to dataset configuration file (default: config/dataset.yaml)'
    )
    
    parser.add_argument(
        '--model',
        default='yolov8n.pt',
        help='YOLO model size: n/s/m/l/x (default: yolov8n.pt)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs (default: 50)'
    )
    
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Image size for training (default: 640)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size (default: 16)'
    )
    
    parser.add_argument(
        '--device',
        default='0',
        help='CUDA device id (0 for GPU, or CPU) (default: 0)'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='Initial learning rate (default: 0.001)'
    )
    
    parser.add_argument(
        '--patience',
        type=int,
        default=10,
        help='Early stopping patience (default: 10)'
    )
    
    parser.add_argument(
        '--output',
        default='models/yolo_finetuned',
        help='Output directory for trained models'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("YOLO FINE-TUNING FOR VEHICLE DETECTION")
    print("=" * 80)
    
    # Check if dataset file exists
    dataset_path = Path(args.data)
    if not dataset_path.exists():
        print(f"⚠️  Dataset configuration file not found: {args.data}")
        print("Creating a template dataset.yaml...")
        print("\nPlease configure your dataset with the following structure:")
        print("""
path: /path/to/dataset  # Dataset root directory
train: images/train     # Train images folder
val: images/val        # Validation images folder
test: images/test      # Test images folder

nc: 4                   # Number of classes
names:                  # Class names
  0: car
  1: motorcycle
  2: bus
  3: truck
        """)
        sys.exit(1)
    
    # Load and train model
    print(f"\nLoading YOLO model: {args.model}")
    model = YOLO(args.model)
    
    print(f"Starting training...")
    print(f"- Epochs: {args.epochs}")
    print(f"- Batch size: {args.batch_size}")
    print(f"- Image size: {args.imgsz}")
    print(f"- Learning rate: {args.lr}")
    print(f"- Device: {args.device}")
    
    # Train
    results = model.train(
        data=str(dataset_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch_size,
        device=args.device,
        lr0=args.lr,
        patience=args.patience,
        project=args.output,
        name='train_results',
        verbose=True,
        save=True,
        plots=True,
        augment=True,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        confidence=0.5
    )
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETED")
    print("=" * 80)
    
    # Evaluate
    print("\nEvaluating model on validation set...")
    metrics = model.val()
    
    print(f"\nValidation Results:")
    print(f"- mAP50: {metrics.box.map50:.4f}")
    print(f"- mAP50-95: {metrics.box.map:.4f}")
    
    # Save best model
    best_model_path = Path(args.output) / "train_results" / "weights" / "best.pt"
    if best_model_path.exists():
        print(f"\n✅ Best model saved to: {best_model_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
