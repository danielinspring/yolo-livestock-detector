"""
Data Splitting Script
Splits preprocessed data into train/validation/test sets for YOLO training
"""

import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm
import argparse


class DatasetSplitter:
    def __init__(self, data_dir="data/processed", train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42):
        """
        Initialize dataset splitter

        Args:
            data_dir: Directory containing images and labels
            train_ratio: Ratio for training set (default: 0.7)
            val_ratio: Ratio for validation set (default: 0.2)
            test_ratio: Ratio for test set (default: 0.1)
            seed: Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

        # Validate ratios
        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Ratios must sum to 1.0, got {total}")

        random.seed(seed)

    def split_dataset(self):
        """Split dataset into train/val/test sets"""

        images_dir = self.data_dir / "images"
        labels_dir = self.data_dir / "labels"

        if not images_dir.exists() or not labels_dir.exists():
            print(f"ERROR: Images or labels directory not found in {self.data_dir}")
            return False

        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']
        image_files = []
        for ext in image_extensions:
            image_files.extend(images_dir.glob(f"*{ext}"))

        # Filter to only include images that have corresponding labels
        valid_pairs = []
        for img_file in image_files:
            label_file = labels_dir / f"{img_file.stem}.txt"
            if label_file.exists():
                valid_pairs.append((img_file, label_file))

        if not valid_pairs:
            print("ERROR: No valid image-label pairs found!")
            return False

        print(f"Found {len(valid_pairs)} valid image-label pairs")

        # Shuffle the data
        random.shuffle(valid_pairs)

        # Calculate split indices
        total = len(valid_pairs)
        train_end = int(total * self.train_ratio)
        val_end = train_end + int(total * self.val_ratio)

        splits = {
            'train': valid_pairs[:train_end],
            'val': valid_pairs[train_end:val_end],
            'test': valid_pairs[val_end:]
        }

        print("\nDataset split:")
        print(f"  Train: {len(splits['train'])} samples ({len(splits['train'])/total*100:.1f}%)")
        print(f"  Val:   {len(splits['val'])} samples ({len(splits['val'])/total*100:.1f}%)")
        print(f"  Test:  {len(splits['test'])} samples ({len(splits['test'])/total*100:.1f}%)")

        # Create directories and copy files
        for split_name, pairs in splits.items():
            split_images_dir = self.data_dir / "images" / split_name
            split_labels_dir = self.data_dir / "labels" / split_name

            split_images_dir.mkdir(parents=True, exist_ok=True)
            split_labels_dir.mkdir(parents=True, exist_ok=True)

            print(f"\nCopying {split_name} set...")
            for img_file, label_file in tqdm(pairs):
                # Copy image
                dest_img = split_images_dir / img_file.name
                if not dest_img.exists():
                    shutil.copy2(img_file, dest_img)

                # Copy label
                dest_label = split_labels_dir / label_file.name
                if not dest_label.exists():
                    shutil.copy2(label_file, dest_label)

        # Clean up original images and labels directories
        print("\nCleaning up original directories...")
        for img_file, _ in valid_pairs:
            img_file.unlink(missing_ok=True)
        for _, label_file in valid_pairs:
            label_file.unlink(missing_ok=True)

        print("\n" + "=" * 60)
        print("Data split complete!")
        print("=" * 60)
        print(f"\nDataset structure:")
        print(f"  {self.data_dir}/")
        print(f"    images/")
        print(f"      train/ ({len(splits['train'])} images)")
        print(f"      val/   ({len(splits['val'])} images)")
        print(f"      test/  ({len(splits['test'])} images)")
        print(f"    labels/")
        print(f"      train/ ({len(splits['train'])} labels)")
        print(f"      val/   ({len(splits['val'])} labels)")
        print(f"      test/  ({len(splits['test'])} labels)")

        # Analyze class distribution
        self.analyze_distribution(splits)

        return True

    def analyze_distribution(self, splits):
        """Analyze class distribution across splits"""
        print("\nClass distribution analysis:")

        for split_name, pairs in splits.items():
            class_counts = {0: 0, 1: 0}  # ride: 0, cowtail: 1
            total_objects = 0

            labels_dir = self.data_dir / "labels" / split_name

            for _, label_file in pairs:
                label_path = labels_dir / label_file.name
                with open(label_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            class_id = int(line.split()[0])
                            class_counts[class_id] = class_counts.get(class_id, 0) + 1
                            total_objects += 1

            print(f"\n  {split_name.upper()}:")
            print(f"    Total objects: {total_objects}")
            print(f"    Ride: {class_counts.get(0, 0)} ({class_counts.get(0, 0)/total_objects*100:.1f}%)")
            print(f"    Cowtail: {class_counts.get(1, 0)} ({class_counts.get(1, 0)/total_objects*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='Split dataset into train/val/test sets')
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/processed',
        help='Directory containing processed data'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.7,
        help='Training set ratio (default: 0.7)'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.2,
        help='Validation set ratio (default: 0.2)'
    )
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.1,
        help='Test set ratio (default: 0.1)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    args = parser.parse_args()

    splitter = DatasetSplitter(
        data_dir=args.data_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )

    splitter.split_dataset()


if __name__ == "__main__":
    main()
