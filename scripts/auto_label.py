"""
Auto-Labeling Script for YOLO Combined Model
Automatically label new images using trained model
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
import cv2
import json


class AutoLabeler:
    def __init__(
        self,
        weights_path,
        img_size=(640, 384),
        conf_threshold=0.25,
        iou_threshold=0.45,
        device='',
    ):
        """
        Initialize auto-labeler

        Args:
            weights_path: Path to model weights
            img_size: Input image size (width, height)
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
            device: Device to use ('' for auto, 'cpu', '0', etc.)
        """
        self.weights_path = Path(weights_path)
        self.img_size = img_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device

        if not self.weights_path.exists():
            raise FileNotFoundError(f"Model weights not found: {weights_path}")

        # Load model
        print(f"Loading model from: {self.weights_path}")
        self.model = YOLO(str(self.weights_path))
        print("Model loaded successfully!")

        # Class names
        self.class_names = ['ride', 'cowtail']

    def label_images(self, source_dir, output_dir, save_images=False, save_visualizations=False):
        """
        Auto-label images in a directory

        Args:
            source_dir: Directory containing images to label
            output_dir: Directory to save labels
            save_images: Whether to copy images to output directory
            save_visualizations: Whether to save visualization images
        """
        source_dir = Path(source_dir)
        output_dir = Path(output_dir)

        if not source_dir.exists():
            raise FileNotFoundError(f"Source directory not found: {source_dir}")

        # Create output directories
        labels_dir = output_dir / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)

        if save_images:
            images_dir = output_dir / "images"
            images_dir.mkdir(parents=True, exist_ok=True)

        if save_visualizations:
            viz_dir = output_dir / "visualizations"
            viz_dir.mkdir(parents=True, exist_ok=True)

        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']
        image_files = []
        for ext in image_extensions:
            image_files.extend(source_dir.glob(f"*{ext}"))

        if not image_files:
            print(f"No images found in {source_dir}")
            return

        print(f"\nFound {len(image_files)} images")
        print(f"Confidence threshold: {self.conf_threshold}")
        print(f"IoU threshold: {self.iou_threshold}")

        # Process images
        total_detections = 0
        class_counts = {0: 0, 1: 0}
        labeled_images = 0
        empty_images = 0

        stats = {
            'total_images': len(image_files),
            'labeled_images': 0,
            'empty_images': 0,
            'total_detections': 0,
            'class_counts': {name: 0 for name in self.class_names},
            'confidence_threshold': self.conf_threshold,
            'iou_threshold': self.iou_threshold,
        }

        print("\nProcessing images...")

        for img_file in tqdm(image_files):
            # Read image to get dimensions
            img = cv2.imread(str(img_file))
            if img is None:
                print(f"Warning: Failed to read {img_file}")
                continue

            img_height, img_width = img.shape[:2]

            # Run inference
            results = self.model.predict(
                img_file,
                imgsz=self.img_size,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False,
            )

            # Process detections
            annotations = []

            if results[0].boxes is not None and len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    # Get box coordinates (xyxy format)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    class_id = int(box.cls[0])

                    # Convert to YOLO format (normalized xywh)
                    x_center = ((x1 + x2) / 2) / img_width
                    y_center = ((y1 + y2) / 2) / img_height
                    width = (x2 - x1) / img_width
                    height = (y2 - y1) / img_height

                    # Create YOLO annotation line
                    annotation = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                    annotations.append(annotation)

                    # Update stats
                    total_detections += 1
                    class_counts[class_id] = class_counts.get(class_id, 0) + 1

            # Save label file
            label_file = labels_dir / f"{img_file.stem}.txt"

            if annotations:
                with open(label_file, 'w') as f:
                    f.write('\n'.join(annotations) + '\n')
                labeled_images += 1
            else:
                # Create empty label file
                label_file.touch()
                empty_images += 1

            # Copy image if requested
            if save_images:
                import shutil
                dest_img = images_dir / img_file.name
                shutil.copy2(img_file, dest_img)

            # Save visualization if requested
            if save_visualizations and annotations:
                viz_img = img.copy()

                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    class_id = int(box.cls[0])

                    # Draw box
                    color = (0, 255, 0) if class_id == 0 else (255, 0, 0)
                    cv2.rectangle(viz_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                    # Draw label
                    label = f"{self.class_names[class_id]}: {conf:.2f}"
                    cv2.putText(viz_img, label, (int(x1), int(y1) - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                viz_file = viz_dir / img_file.name
                cv2.imwrite(str(viz_file), viz_img)

        # Update stats
        stats['labeled_images'] = labeled_images
        stats['empty_images'] = empty_images
        stats['total_detections'] = total_detections
        stats['class_counts'] = {
            'ride': class_counts.get(0, 0),
            'cowtail': class_counts.get(1, 0),
        }

        # Save stats
        stats_file = output_dir / 'labeling_stats.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        # Print summary
        print("\n" + "=" * 60)
        print("Auto-Labeling Complete!")
        print("=" * 60)
        print(f"\nStatistics:")
        print(f"  Total images: {len(image_files)}")
        print(f"  Images with detections: {labeled_images}")
        print(f"  Images without detections: {empty_images}")
        print(f"  Total detections: {total_detections}")
        print(f"  Ride detections: {class_counts.get(0, 0)}")
        print(f"  Cowtail detections: {class_counts.get(1, 0)}")
        print(f"\nOutput:")
        print(f"  Labels: {labels_dir}")
        if save_images:
            print(f"  Images: {images_dir}")
        if save_visualizations:
            print(f"  Visualizations: {viz_dir}")
        print(f"  Stats: {stats_file}")

        return stats

    def label_single_image(self, image_path, output_path=None, visualize=False):
        """
        Label a single image

        Args:
            image_path: Path to image
            output_path: Path to save label (optional)
            visualize: Whether to save visualization
        """
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Failed to read image: {image_path}")

        img_height, img_width = img.shape[:2]

        # Run inference
        results = self.model.predict(
            image_path,
            imgsz=self.img_size,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        # Process detections
        annotations = []

        if results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                class_id = int(box.cls[0])

                # Convert to YOLO format
                x_center = ((x1 + x2) / 2) / img_width
                y_center = ((y1 + y2) / 2) / img_height
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height

                annotation = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                annotations.append(annotation)

        # Save label
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w') as f:
                if annotations:
                    f.write('\n'.join(annotations) + '\n')

            print(f"Label saved to: {output_path}")

        # Print results
        print(f"\nDetections: {len(annotations)}")
        for ann in annotations:
            parts = ann.split()
            class_id = int(parts[0])
            print(f"  {self.class_names[class_id]}")

        return annotations


def main():
    parser = argparse.ArgumentParser(description='Auto-label images using trained YOLO model')

    parser.add_argument(
        '--weights',
        type=str,
        default='models/best.pt',
        help='Path to model weights'
    )
    parser.add_argument(
        '--source',
        type=str,
        required=True,
        help='Source directory containing images'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for labels'
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
        '--conf',
        type=float,
        default=0.25,
        help='Confidence threshold (default: 0.25)'
    )
    parser.add_argument(
        '--iou',
        type=float,
        default=0.45,
        help='IoU threshold for NMS (default: 0.45)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='',
        help='Device to use (default: auto, options: cpu, 0, 0,1, etc.)'
    )
    parser.add_argument(
        '--save-images',
        action='store_true',
        help='Copy images to output directory'
    )
    parser.add_argument(
        '--save-viz',
        action='store_true',
        help='Save visualization images'
    )

    args = parser.parse_args()

    # Initialize auto-labeler
    labeler = AutoLabeler(
        weights_path=args.weights,
        img_size=(args.img_width, args.img_height),
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        device=args.device,
    )

    # Run auto-labeling
    labeler.label_images(
        source_dir=args.source,
        output_dir=args.output,
        save_images=args.save_images,
        save_visualizations=args.save_viz,
    )


if __name__ == "__main__":
    main()
