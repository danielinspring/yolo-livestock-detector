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
from tqdm import tqdm
import yaml


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

        self.class_names = ["ride", "cowtail"]  # Final classes

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
            'names': ['ride', 'cowtail']
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

    def process(self, source_images_dir=None, background_ratio=0.0):
        """
        Run full preprocessing pipeline

        Args:
            source_images_dir: Optional path to images directory
            background_ratio: Ratio of background images to include (0.0 to 1.0)
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
        default='data/processed',
        help='Output directory for processed data'
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
        default=0.0,
        help='Ratio of background images to include (0.0 to 1.0, default: 0.0)'
    )

    args = parser.parse_args()

    preprocessor = LabelStudioPreprocessor(args.input, args.output)
    preprocessor.process(source_images_dir=args.images, background_ratio=args.background_ratio)


if __name__ == "__main__":
    main()
