"""
Training Script for YOLO Combined Model
Trains YOLOv8/v11 model on ride and cowtail dataset
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import yaml
import torch


class YOLOTrainer:
    def __init__(
        self,
        model_version='yolov8s',
        data_config='configs/dataset.yaml',
        project='results/train',
        name='combined_model',
        img_size=(640, 384),
        epochs=100,
        batch_size=16,
        device='',
    ):
        """
        Initialize YOLO trainer

        Args:
            model_version: YOLO model version (yolov8s, yolov8n, yolov11s, etc.)
            data_config: Path to dataset YAML config
            project: Project directory for results
            name: Experiment name
            img_size: Input image size (width, height)
            epochs: Number of training epochs
            batch_size: Batch size
            device: Device to use ('' for auto, 'cpu', '0', '0,1', etc.)
        """
        self.model_version = model_version
        self.data_config = data_config
        self.project = project
        self.name = name
        self.img_size = img_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device

        # Check if GPU is available
        self.gpu_available = torch.cuda.is_available()
        print(f"\nGPU Available: {self.gpu_available}")
        if self.gpu_available:
            print(f"GPU Device: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    def verify_data_config(self):
        """Verify dataset configuration exists and is valid"""
        config_path = Path(self.data_config)

        if not config_path.exists():
            raise FileNotFoundError(f"Dataset config not found: {self.data_config}")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        print("\nDataset Configuration:")
        print(f"  Path: {config.get('path', 'N/A')}")
        print(f"  Classes: {config.get('nc', 'N/A')}")
        print(f"  Names: {config.get('names', 'N/A')}")

        # Check if data directories exist
        data_path = Path(config.get('path', ''))
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {data_path}")

        train_path = data_path / config.get('train', '')
        val_path = data_path / config.get('val', '')

        if not train_path.exists():
            raise FileNotFoundError(f"Training images not found: {train_path}")
        if not val_path.exists():
            raise FileNotFoundError(f"Validation images not found: {val_path}")

        # Count images
        train_images = len(list(train_path.glob("*.[jJ][pP][gG]"))) + \
                      len(list(train_path.glob("*.[pP][nN][gG]")))
        val_images = len(list(val_path.glob("*.[jJ][pP][gG]"))) + \
                    len(list(val_path.glob("*.[pP][nN][gG]")))

        print(f"\nDataset Statistics:")
        print(f"  Training images: {train_images}")
        print(f"  Validation images: {val_images}")

        return config

    def train(self, resume=False, pretrained=True):
        """
        Train YOLO model

        Args:
            resume: Resume training from last checkpoint
            pretrained: Use pretrained weights
        """
        print("\n" + "=" * 60)
        print("YOLO Training")
        print("=" * 60)

        # Verify data configuration
        self.verify_data_config()

        # Load model
        if resume:
            # Resume from last checkpoint
            checkpoint_path = Path(self.project) / self.name / "weights" / "last.pt"
            if checkpoint_path.exists():
                print(f"\nResuming from checkpoint: {checkpoint_path}")
                model = YOLO(str(checkpoint_path))
            else:
                print(f"\nCheckpoint not found: {checkpoint_path}")
                print("Starting fresh training...")
                model = YOLO(f"{self.model_version}.pt" if pretrained else f"{self.model_version}.yaml")
        else:
            # Start fresh training
            if pretrained:
                print(f"\nLoading pretrained model: {self.model_version}.pt")
                model = YOLO(f"{self.model_version}.pt")
            else:
                print(f"\nInitializing model from scratch: {self.model_version}.yaml")
                model = YOLO(f"{self.model_version}.yaml")

        print(f"\nTraining Configuration:")
        print(f"  Model: {self.model_version}")
        print(f"  Image size: {self.img_size}")
        print(f"  Epochs: {self.epochs}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Device: {self.device if self.device else 'auto'}")
        print(f"  Pretrained: {pretrained}")

        # Train model
        print("\nStarting training...\n")

        results = model.train(
            data=self.data_config,
            epochs=self.epochs,
            imgsz=self.img_size,
            batch=self.batch_size,
            device=self.device,
            project=self.project,
            name=self.name,
            exist_ok=True,
            pretrained=pretrained,
            optimizer='auto',
            verbose=True,
            seed=42,
            deterministic=True,
            # Data augmentation
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=0.0,
            translate=0.1,
            scale=0.5,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.0,
            copy_paste=0.0,
        )

        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)

        # Get the actual save directory from training results
        # YOLO saves to runs/detect/{project}/{name}, not directly to {project}/{name}
        save_dir = Path(results.save_dir)
        
        # Display results
        best_weights = save_dir / "weights" / "best.pt"
        last_weights = save_dir / "weights" / "last.pt"

        print(f"\nModel weights saved:")
        print(f"  Best: {best_weights}")
        print(f"  Last: {last_weights}")

        print(f"\nResults saved to: {save_dir}")

        # Copy best model to models/{date} directory
        import shutil
        from datetime import datetime
        today_date = datetime.now().strftime("%Y-%m-%d")
        models_dir = Path("models") / today_date
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename with timestamp
        base_name = self.name  # Use experiment name as base filename
        timestamp = datetime.now().strftime("%H%M%S")
        final_model = models_dir / f"{base_name}_{timestamp}.pt"
        
        # Check which weights file to copy (best.pt preferred, fall back to last.pt)
        if best_weights.exists():
            shutil.copy2(best_weights, final_model)
            print(f"\nBest model copied to: {final_model}")
        elif last_weights.exists():
            shutil.copy2(last_weights, final_model)
            print(f"\nNote: best.pt not found, copied last.pt to: {final_model}")
        else:
            print(f"\nWarning: No model weights found to copy. Check training output.")

        return results


def main():
    parser = argparse.ArgumentParser(description='Train YOLO model on ride and cowtail dataset')

    parser.add_argument(
        '--model',
        type=str,
        default='yolov8s',
        choices=['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x',
                 'yolo11n', 'yolo11s', 'yolo11m', 'yolo11l', 'yolo11x'],
        help='YOLO model version (default: yolov8s)'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='configs/dataset.yaml',
        help='Path to dataset YAML config'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs (default: 100)'
    )
    parser.add_argument(
        '--batch',
        type=int,
        default=16,
        help='Batch size (default: 16)'
    )
    parser.add_argument(
        '--img-width',
        type=int,
        default=640,
        help='Image width (default: 640)'
    )
    parser.add_argument(
        '--img-height',
        type=int,
        default=384,
        help='Image height (default: 384)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='',
        help='Device to use (default: auto, options: cpu, 0, 0,1, etc.)'
    )
    parser.add_argument(
        '--project',
        type=str,
        default='results/train',
        help='Project directory for results'
    )
    parser.add_argument(
        '--name',
        type=str,
        default='combined_model',
        help='Experiment name'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume training from last checkpoint'
    )
    parser.add_argument(
        '--no-pretrained',
        action='store_true',
        help='Train from scratch without pretrained weights'
    )

    args = parser.parse_args()

    trainer = YOLOTrainer(
        model_version=args.model,
        data_config=args.data,
        project=args.project,
        name=args.name,
        img_size=(args.img_width, args.img_height),
        epochs=args.epochs,
        batch_size=args.batch,
        device=args.device,
    )

    trainer.train(resume=args.resume, pretrained=not args.no_pretrained)


if __name__ == "__main__":
    main()
