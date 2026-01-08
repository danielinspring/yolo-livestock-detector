"""
Data Preprocessing Script for YOLO Combined Model
Filters Label Studio export to keep only 'ride' and 'cowtail' classes
and remaps them to 0 and 1 respectively.
"""

import os
import shutil
import json
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
            'path': str(self.output_dir.absolute()),
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

    def copy_images(self, source_images_dir=None):
        """
        Copy images to processed directory

        Args:
            source_images_dir: Optional path to images if not in Label Studio export
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

        all_images = []
        for ext in image_extensions:
            all_images.extend(images_dir.glob(f"*{ext}"))

        print(f"\nCopying images with ride/cowtail annotations...")
        for img_file in tqdm(all_images):
            # Check if this image has a corresponding label
            if img_file.stem in kept_labels:
                dest = output_images_dir / img_file.name
                if not dest.exists():
                    shutil.copy2(img_file, dest)
                copied += 1

        print(f"Copied {copied} images")
        return copied

    def process(self, source_images_dir=None):
        """
        Run full preprocessing pipeline

        Args:
            source_images_dir: Optional path to images directory
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
            self.copy_images(source_images_dir)

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
        default='data/project-8-at-2026-01-07-07-09-0780865d',
        help='Input Label Studio export directory'
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

    args = parser.parse_args()

    preprocessor = LabelStudioPreprocessor(args.input, args.output)
    preprocessor.process(source_images_dir=args.images)


if __name__ == "__main__":
    main()
