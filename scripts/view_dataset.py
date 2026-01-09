"""
FiftyOne Dataset Viewer for YOLO Processed Data
Launches FiftyOne app to visualize images and annotations.
"""

import argparse
from pathlib import Path

try:
    import fiftyone as fo
except ImportError:
    print("FiftyOne not installed. Installing...")
    import subprocess
    subprocess.check_call(["pip", "install", "fiftyone"])
    import fiftyone as fo


def load_yolo_dataset(data_dir: str, name: str = "yolo_dataset", split: str = None):
    """
    Load YOLO format dataset into FiftyOne
    
    Args:
        data_dir: Path to data directory containing images/ and labels/
        name: Dataset name in FiftyOne
        split: Optional split to load (train, val, test). If None, loads all splits or flat structure.
    """
    data_path = Path(data_dir)
    images_dir = data_path / "images"
    labels_dir = data_path / "labels"
    
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")
    
    # Class names for the combined model
    classes = ["ride", "cowtail"]
    
    # Delete existing dataset with same name if exists
    if fo.dataset_exists(name):
        fo.delete_dataset(name)
    
    print(f"Loading dataset from {data_path}...")
    
    # Create dataset
    dataset = fo.Dataset(name)
    
    # Check if data is split into train/val/test subdirectories
    has_splits = (images_dir / "train").exists() or (images_dir / "val").exists()
    
    if has_splits:
        # Load from split directories
        splits = ["train", "val", "test"] if split is None else [split]
        for s in splits:
            img_split_dir = images_dir / s
            lbl_split_dir = labels_dir / s
            if img_split_dir.exists():
                print(f"  Loading {s} split...")
                samples = load_samples_from_dir(img_split_dir, lbl_split_dir, classes, split_tag=s)
                dataset.add_samples(samples)
    else:
        # Load from flat structure
        print(f"  Images: {images_dir}")
        print(f"  Labels: {labels_dir}")
        samples = load_samples_from_dir(images_dir, labels_dir, classes)
        dataset.add_samples(samples)
    
    print(f"\nDataset loaded: {len(dataset)} samples")
    print(f"Classes: {classes}")
    
    return dataset


def load_samples_from_dir(images_dir: Path, labels_dir: Path, classes: list, split_tag: str = None):
    """Load samples from a directory pair of images and labels."""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(images_dir.glob(f"*{ext}"))
        image_files.extend(images_dir.glob(f"*{ext.upper()}"))
    
    samples = []
    for img_path in image_files:
        # Find corresponding label file
        label_path = labels_dir / f"{img_path.stem}.txt"
        
        sample = fo.Sample(filepath=str(img_path.absolute()))
        
        # Add split tag if provided
        if split_tag:
            sample.tags.append(split_tag)
        
        if label_path.exists():
            # Parse YOLO labels
            detections = []
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # Convert YOLO format (center x, center y, w, h) to FiftyOne format (x, y, w, h)
                        # FiftyOne uses top-left corner
                        x = x_center - width / 2
                        y = y_center - height / 2
                        
                        label = classes[class_id] if class_id < len(classes) else f"class_{class_id}"
                        
                        detections.append(
                            fo.Detection(
                                label=label,
                                bounding_box=[x, y, width, height],
                            )
                        )
            
            sample["ground_truth"] = fo.Detections(detections=detections)
        
        samples.append(sample)
    
    return samples


def main():
    parser = argparse.ArgumentParser(description='View YOLO dataset with FiftyOne')
    parser.add_argument(
        '--data',
        type=str,
        default='data/processed',
        help='Path to data directory (default: data/processed)'
    )
    parser.add_argument(
        '--name',
        type=str,
        default='yolo_processed',
        help='Dataset name in FiftyOne'
    )
    parser.add_argument(
        '--split',
        type=str,
        choices=['train', 'val', 'test'],
        default=None,
        help='Load specific split only (default: all splits)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=5151,
        help='Port for FiftyOne app'
    )
    parser.add_argument(
        '--desktop',
        action='store_true',
        help='Launch as desktop app (may fix some browser issues)'
    )
    
    args = parser.parse_args()
    
    # Load dataset
    dataset = load_yolo_dataset(args.data, args.name, args.split)
    
    # Launch FiftyOne app
    print(f"\nLaunching FiftyOne app on port {args.port}...")
    print("Press Ctrl+C to stop\n")
    
    session = fo.launch_app(dataset, port=args.port)
    session.wait()


if __name__ == "__main__":
    main()
