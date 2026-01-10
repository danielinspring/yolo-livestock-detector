"""
Data Preprocessing Script for YOLO Combined Model
Filters Label Studio export to keep only 'ride' and 'cowtail' classes
and remaps them to 0 and 1 respectively.
"""

import os
import shutil
import json
import random
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import yaml
import cv2
import numpy as np

try:
    import albumentations as A
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False


class LabelStudioPreprocessor:
    def __init__(self, input_dir, output_dir="data/processed"):
        """
        Initialize preprocessor

        Args:
            input_dir: Path to Label Studio export directory
            output_dir: Path to output processed data
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)

        # Original class mapping from Label Studio
        self.original_classes = {
            0: "cowtail",
            8: "ride"
        }

        # New class mapping for combined model
        self.new_class_mapping = {
            0: 0,  # cowtail stays 0
            8: 1   # ride becomes 1
        }

        self.class_names = ["cowtail", "ride"]  # Final classes (0=cowtail, 1=ride)

        # Augmentation transforms for class balancing
        self.augmentations = self._setup_augmentations() if ALBUMENTATIONS_AVAILABLE else None

    def _setup_augmentations(self):
        """Setup augmentation transforms with bbox support"""
        return {
            'flip': A.Compose([
                A.HorizontalFlip(p=1.0)
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])),

            'bright': A.Compose([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0, p=1.0)
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])),

            'rot': A.Compose([
                A.Rotate(limit=15, p=1.0, border_mode=cv2.BORDER_REFLECT_101)
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])),

            'noise': A.Compose([
                A.GaussNoise(var_limit=(10, 50), p=1.0)
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])),
        }

    def _parse_yolo_label(self, label_path):
        """Parse YOLO format label file"""
        bboxes = []
        class_labels = []

        if not label_path.exists():
            return bboxes, class_labels

        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:5])
                    bboxes.append([x_center, y_center, width, height])
                    class_labels.append(class_id)

        return bboxes, class_labels

    def _save_yolo_label(self, label_path, bboxes, class_labels):
        """Save bboxes in YOLO format"""
        with open(label_path, 'w') as f:
            for bbox, class_id in zip(bboxes, class_labels):
                x_center, y_center, width, height = bbox
                # Ensure class_id is integer
                f.write(f"{int(class_id)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    def _apply_augmentation(self, image, bboxes, class_labels, aug_name):
        """Apply a single augmentation and return transformed image and bboxes"""
        if not self.augmentations or aug_name not in self.augmentations:
            return None, None, None

        transform = self.augmentations[aug_name]

        # Handle empty bboxes
        if len(bboxes) == 0:
            transformed = transform(image=image, bboxes=[], class_labels=[])
            return transformed['image'], [], []

        try:
            transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
            return transformed['image'], transformed['bboxes'], transformed['class_labels']
        except Exception as e:
            print(f"    Warning: Augmentation {aug_name} failed: {e}")
            return None, None, None

    def augment_minority_class(self, target_ratio=2.0, max_ratio=2.5):
        """
        Augment minority class (cowtail) images to balance dataset.
        Augmentation only happens if current ratio exceeds max_ratio.
        Target ratio is ride:cowtail (e.g., 2.0 means 2:1 ratio)

        Args:
            target_ratio: Target ratio of ride:cowtail (default 2.0 for 2:1)
            max_ratio: Maximum acceptable ratio before augmentation (default 2.5)
        """
        if not ALBUMENTATIONS_AVAILABLE:
            print("\nWARNING: Albumentations not installed. Skipping augmentation.")
            print("Install with: pip install albumentations")
            return

        output_labels_dir = self.output_dir / "labels"
        output_images_dir = self.output_dir / "images"

        # Count current class distribution
        ride_count = 0
        cowtail_count = 0
        cowtail_images = []  # List of (image_path, label_path) for cowtail images

        print("\nAnalyzing class distribution...")
        for label_file in output_labels_dir.glob("*.txt"):
            _, class_labels = self._parse_yolo_label(label_file)

            ride_count += class_labels.count(1)
            cowtail_count += class_labels.count(0)

            # Find corresponding image if it has cowtail annotations
            if 0 in class_labels:  # cowtail is class 0
                img_stem = label_file.stem
                for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                    img_path = output_images_dir / f"{img_stem}{ext}"
                    if img_path.exists():
                        cowtail_images.append((img_path, label_file))
                        break

        print(f"  Current distribution:")
        print(f"    Ride: {ride_count}")
        print(f"    Cowtail: {cowtail_count}")

        if cowtail_count == 0:
            print("  No cowtail annotations found. Skipping augmentation.")
            return

        current_ratio = ride_count / cowtail_count if cowtail_count > 0 else float('inf')
        print(f"    Current ratio (ride:cowtail): {current_ratio:.2f}:1")

        # Only augment if ratio exceeds max_ratio (2.5:1)
        if current_ratio <= max_ratio:
            print(f"  Already balanced (ratio {current_ratio:.2f}:1 <= {max_ratio}:1). Skipping augmentation.")
            return

        # Calculate how many cowtail annotations we need
        target_cowtail = int(ride_count / target_ratio)
        needed_cowtail = target_cowtail - cowtail_count

        if needed_cowtail <= 0:
            print("  No augmentation needed.")
            return

        print(f"\n  Target cowtail count: {target_cowtail}")
        print(f"  Need to generate: {needed_cowtail} more cowtail annotations")

        # Augmentation suffixes
        aug_types = ['flip', 'bright', 'rot', 'noise']

        # Calculate how many times to augment each image
        # Each image can generate up to 4 augmented versions
        augmented_count = 0
        aug_round = 0

        print(f"\nAugmenting {len(cowtail_images)} cowtail images...")

        while augmented_count < needed_cowtail and aug_round < len(aug_types):
            aug_name = aug_types[aug_round]
            print(f"\n  Applying {aug_name} augmentation...")

            for img_path, label_path in tqdm(cowtail_images, desc=f"  {aug_name}"):
                if augmented_count >= needed_cowtail:
                    break

                # Read image
                image = cv2.imread(str(img_path))
                if image is None:
                    continue
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Parse labels
                bboxes, class_labels = self._parse_yolo_label(label_path)

                # Apply augmentation
                aug_image, aug_bboxes, aug_class_labels = self._apply_augmentation(
                    image, bboxes, class_labels, aug_name
                )

                if aug_image is None:
                    continue

                # Save augmented image
                aug_img_name = f"{img_path.stem}_{aug_name}{img_path.suffix}"
                aug_img_path = output_images_dir / aug_img_name
                aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(aug_img_path), aug_image_bgr)

                # Save augmented labels
                aug_label_name = f"{label_path.stem}_{aug_name}.txt"
                aug_label_path = output_labels_dir / aug_label_name
                self._save_yolo_label(aug_label_path, aug_bboxes, aug_class_labels)

                # Count new cowtail annotations
                new_cowtail = aug_class_labels.count(0) if isinstance(aug_class_labels, list) else list(aug_class_labels).count(0)
                augmented_count += new_cowtail

            aug_round += 1

        # Final count
        final_cowtail = cowtail_count + augmented_count
        final_ratio = ride_count / final_cowtail if final_cowtail > 0 else float('inf')

        print(f"\nAugmentation Summary:")
        print(f"  Generated {augmented_count} new cowtail annotations")
        print(f"  Final distribution:")
        print(f"    Ride: {ride_count}")
        print(f"    Cowtail: {final_cowtail}")
        print(f"    Final ratio (ride:cowtail): {final_ratio:.2f}:1")

    def load_notes(self):
        """Load notes.json to understand class structure"""
        notes_path = self.input_dir / "notes.json"
        if notes_path.exists():
            with open(notes_path, 'r') as f:
                notes = json.load(f)
            print("Available categories:")
            for cat in notes.get('categories', []):
                print(f"  ID {cat['id']}: {cat['name']}")
            return notes
        return None

    def filter_and_remap_labels(self):
        """
        Filter labels to keep only ride and cowtail,
        and remap class IDs to 0 and 1
        """
        labels_dir = self.input_dir / "labels"
        output_labels_dir = self.output_dir / "labels"
        output_labels_dir.mkdir(parents=True, exist_ok=True)

        total_files = 0
        processed_files = 0
        total_annotations = 0
        kept_annotations = 0

        label_files = list(labels_dir.glob("*.txt"))

        print(f"\nProcessing {len(label_files)} label files...")

        for label_file in tqdm(label_files):
            total_files += 1
            new_annotations = []

            with open(label_file, 'r') as f:
                lines = f.readlines()

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                total_annotations += 1
                parts = line.split()

                if len(parts) < 5:
                    continue

                class_id = int(parts[0])

                # Keep only ride and cowtail
                if class_id in self.new_class_mapping:
                    # Remap class ID
                    new_class_id = self.new_class_mapping[class_id]
                    new_line = f"{new_class_id} {' '.join(parts[1:])}"
                    new_annotations.append(new_line)
                    kept_annotations += 1

            # Only save files that have relevant annotations
            if new_annotations:
                output_file = output_labels_dir / label_file.name
                with open(output_file, 'w') as f:
                    f.write('\n'.join(new_annotations) + '\n')
                processed_files += 1

        print(f"\nLabel Filtering Summary:")
        print(f"  Total label files: {total_files}")
        print(f"  Files with ride/cowtail: {processed_files}")
        print(f"  Total annotations: {total_annotations}")
        print(f"  Kept annotations: {kept_annotations}")
        print(f"  Filtered out: {total_annotations - kept_annotations}")

        return processed_files

    def create_dataset_yaml(self):
        """Create YOLO dataset configuration file"""
        yaml_content = {
            'path': str(self.output_dir),  # Use relative path for portability
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': 2,  # number of classes
            'names': ['cowtail', 'ride']
        }

        yaml_path = Path("configs/dataset.yaml")
        yaml_path.parent.mkdir(parents=True, exist_ok=True)

        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)

        print(f"\nCreated dataset config: {yaml_path}")
        return yaml_path

    def check_images(self):
        """Check for image files"""
        images_dir = self.input_dir / "images"

        if not images_dir.exists():
            print(f"\nWARNING: Images directory not found at {images_dir}")
            print("Please provide image files in one of these ways:")
            print("  1. Place images in data/images/")
            print("  2. Download from Label Studio")
            print("  3. Provide path to existing images")
            return 0

        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(images_dir.glob(f"*{ext}"))
            image_files.extend(images_dir.glob(f"*{ext.upper()}"))

        print(f"\nFound {len(image_files)} image files")
        return len(image_files)

    def copy_images(self, source_images_dir=None, background_ratio=0.0):
        """
        Copy images to processed directory

        Args:
            source_images_dir: Optional path to images if not in Label Studio export
            background_ratio: Ratio of background images to include (0.0 to 1.0)
        """
        if source_images_dir:
            images_dir = Path(source_images_dir)
        else:
            images_dir = self.input_dir / "images"

        if not images_dir.exists():
            print(f"ERROR: Images directory not found: {images_dir}")
            return 0

        # Get list of label files we kept
        output_labels_dir = self.output_dir / "labels"
        kept_labels = set(f.stem for f in output_labels_dir.glob("*.txt"))

        output_images_dir = self.output_dir / "images"
        output_images_dir.mkdir(parents=True, exist_ok=True)

        # Copy only images that have corresponding labels
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']
        copied = 0
        background_copied = 0

        all_images = []
        for ext in image_extensions:
            all_images.extend(images_dir.glob(f"*{ext}"))

        # Separate images with labels and background images
        images_with_labels = []
        background_images = []

        for img_file in all_images:
            if img_file.stem in kept_labels:
                images_with_labels.append(img_file)
            else:
                background_images.append(img_file)

        print(f"\nCopying images with ride/cowtail annotations...")
        for img_file in tqdm(images_with_labels):
            dest = output_images_dir / img_file.name
            if not dest.exists():
                shutil.copy2(img_file, dest)
            copied += 1

        print(f"Copied {copied} images with annotations")

        # Add background images if ratio > 0
        if background_ratio > 0 and background_images:
            num_background = int(len(images_with_labels) * background_ratio)
            num_background = min(num_background, len(background_images))

            selected_backgrounds = random.sample(background_images, num_background)

            print(f"\nAdding {num_background} background images ({background_ratio*100:.0f}% of labeled images)...")
            for img_file in tqdm(selected_backgrounds):
                dest = output_images_dir / img_file.name
                if not dest.exists():
                    shutil.copy2(img_file, dest)
                # Create empty label file for background image
                empty_label = output_labels_dir / f"{img_file.stem}.txt"
                empty_label.touch()
                background_copied += 1

            print(f"Added {background_copied} background images")

        print(f"\nTotal images: {copied + background_copied}")

        # Clean up orphan labels (labels without matching images)
        self.cleanup_orphan_labels()

        return copied + background_copied

    def cleanup_orphan_labels(self):
        """Remove label files that don't have corresponding images"""
        output_labels_dir = self.output_dir / "labels"
        output_images_dir = self.output_dir / "images"

        # Get all image stems
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']
        image_stems = set()
        for ext in image_extensions:
            for img in output_images_dir.glob(f"*{ext}"):
                image_stems.add(img.stem)

        # Remove orphan labels
        orphan_count = 0
        for label_file in output_labels_dir.glob("*.txt"):
            if label_file.stem not in image_stems:
                label_file.unlink()
                orphan_count += 1

        if orphan_count > 0:
            print(f"\nCleaned up {orphan_count} orphan labels (no matching images)")

    def process(self, source_images_dir=None, background_ratio=0.0, augment=True, target_ratio=2.0):
        """
        Run full preprocessing pipeline

        Args:
            source_images_dir: Optional path to images directory
            background_ratio: Ratio of background images to include (0.0 to 1.0)
            augment: Whether to augment minority class (default True)
            target_ratio: Target ride:cowtail ratio for augmentation (default 2.0)
        """
        print("=" * 60)
        print("YOLO Data Preprocessing")
        print("=" * 60)

        # Load and display notes
        self.load_notes()

        # Filter and remap labels
        processed_files = self.filter_and_remap_labels()

        if processed_files == 0:
            print("\nERROR: No files with ride/cowtail annotations found!")
            return False

        # Check for images
        image_count = self.check_images()

        # Copy images if available
        if image_count > 0 or source_images_dir:
            self.copy_images(source_images_dir, background_ratio=background_ratio)

        # Augment minority class (cowtail) to balance dataset
        if augment:
            self.augment_minority_class(target_ratio=target_ratio)

        # Create dataset YAML
        self.create_dataset_yaml()

        print("\n" + "=" * 60)
        print("Preprocessing complete!")
        print("=" * 60)
        print(f"\nProcessed data location: {self.output_dir}")
        print("\nNext steps:")
        print("  1. If images are missing, provide them using:")
        print("     python scripts/preprocess_data.py --images /path/to/images")
        print("  2. Run data split script:")
        print("     python scripts/split_data.py")
        print("  3. Start training:")
        print("     python scripts/train.py")

        return True


def main():
    import argparse

    # Generate default output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_output = f"data/dataset/processed_{timestamp}"

    parser = argparse.ArgumentParser(description='Preprocess Label Studio data for YOLO')
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to Label Studio export directory (e.g., data/your-export-folder)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=default_output,
        help='Output directory for processed data (default: data/dataset/processed_YYYYMMDD_HHMMSS)'
    )
    parser.add_argument(
        '--images',
        type=str,
        default=None,
        help='Path to images directory if not in Label Studio export'
    )
    parser.add_argument(
        '--background-ratio',
        type=float,
        default=0.065,
        help='Ratio of background images to include (0.0 to 1.0, default: 0.065 = 6.5%%)'
    )
    parser.add_argument(
        '--no-augment',
        action='store_true',
        help='Disable minority class augmentation'
    )
    parser.add_argument(
        '--target-ratio',
        type=float,
        default=2.0,
        help='Target ride:cowtail ratio for augmentation (default: 2.0 for 2:1)'
    )

    args = parser.parse_args()

    preprocessor = LabelStudioPreprocessor(args.input, args.output)
    preprocessor.process(
        source_images_dir=args.images,
        background_ratio=args.background_ratio,
        augment=not args.no_augment,
        target_ratio=args.target_ratio
    )


if __name__ == "__main__":
    main()
