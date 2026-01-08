"""
Setup Checker Script
Verifies project setup and provides guidance on next steps
"""

import sys
from pathlib import Path
import importlib.util


def check_python_version():
    """Check Python version"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"  ✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"  ✗ Python {version.major}.{version.minor}.{version.micro} (requires 3.8+)")
        return False


def check_dependencies():
    """Check if required packages are installed"""
    print("\nChecking dependencies...")
    required_packages = [
        'ultralytics',
        'cv2',
        'numpy',
        'pandas',
        'matplotlib',
        'PIL',
        'yaml',
        'tqdm',
    ]

    all_installed = True
    for package in required_packages:
        package_name = 'opencv-python' if package == 'cv2' else package
        package_name = 'pillow' if package == 'PIL' else package_name
        package_name = 'pyyaml' if package == 'yaml' else package_name

        spec = importlib.util.find_spec(package)
        if spec is not None:
            print(f"  ✓ {package_name}")
        else:
            print(f"  ✗ {package_name} (not installed)")
            all_installed = False

    if not all_installed:
        print("\n  Run: pip install -r requirements.txt")

    return all_installed


def check_data():
    """Check data availability"""
    print("\nChecking data...")

    # Check Label Studio export
    labelstudio_dir = Path("data/project-8-at-2026-01-07-07-09-0780865d")
    if labelstudio_dir.exists():
        print(f"  ✓ Label Studio export found")

        # Check labels
        labels_dir = labelstudio_dir / "labels"
        if labels_dir.exists():
            label_count = len(list(labels_dir.glob("*.txt")))
            print(f"    - {label_count} label files")
        else:
            print(f"    ✗ Labels directory not found")

        # Check images
        images_dir = labelstudio_dir / "images"
        if images_dir.exists():
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']
            image_count = sum(len(list(images_dir.glob(f"*{ext}"))) for ext in image_extensions)
            if image_count > 0:
                print(f"    ✓ {image_count} images found")
            else:
                print(f"    ✗ No images found in images directory")
                print(f"      → Download images from Label Studio")
                print(f"      → Or provide path using: python scripts/preprocess_data.py --images /path/to/images")
        else:
            print(f"    ✗ Images directory not found")
    else:
        print(f"  ✗ Label Studio export not found")
        print(f"    Expected at: {labelstudio_dir}")

    # Check processed data
    processed_dir = Path("data/processed")
    if processed_dir.exists():
        print(f"\n  ✓ Processed data directory exists")

        # Check if data is split
        train_images = processed_dir / "images" / "train"
        if train_images.exists():
            train_count = len(list(train_images.glob("*.jpg"))) + len(list(train_images.glob("*.png")))
            val_images = processed_dir / "images" / "val"
            val_count = len(list(val_images.glob("*.jpg"))) + len(list(val_images.glob("*.png"))) if val_images.exists() else 0
            test_images = processed_dir / "images" / "test"
            test_count = len(list(test_images.glob("*.jpg"))) + len(list(test_images.glob("*.png"))) if test_images.exists() else 0

            print(f"    - Train: {train_count} images")
            print(f"    - Val: {val_count} images")
            print(f"    - Test: {test_count} images")

            if train_count > 0:
                return True
        else:
            print(f"    ✗ Data not split yet")
            print(f"      → Run: python scripts/split_data.py")
    else:
        print(f"\n  ✗ Processed data not found")
        print(f"    → Run: python scripts/preprocess_data.py")

    return False


def check_models():
    """Check for trained models"""
    print("\nChecking models...")

    models_dir = Path("models")
    if models_dir.exists():
        best_model = models_dir / "best.pt"
        if best_model.exists():
            size_mb = best_model.stat().st_size / (1024 * 1024)
            print(f"  ✓ Trained model found: {best_model} ({size_mb:.1f} MB)")
            return True
        else:
            print(f"  ✗ No trained model found")
            print(f"    → Run: python scripts/train.py")
    else:
        print(f"  ✗ Models directory not found")
        print(f"    → Run: python scripts/train.py")

    return False


def check_gpu():
    """Check GPU availability"""
    print("\nChecking GPU...")

    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"  ✓ GPU available: {gpu_name}")
            print(f"    - Memory: {gpu_memory:.1f} GB")
            return True
        else:
            print(f"  ℹ No GPU available (will use CPU)")
            print(f"    Training will be slower on CPU")
            return False
    except ImportError:
        print(f"  ℹ PyTorch not installed, cannot check GPU")
        return False


def suggest_next_steps(data_ready, model_ready):
    """Suggest next steps based on setup status"""
    print("\n" + "=" * 60)
    print("Next Steps:")
    print("=" * 60)

    if not data_ready:
        print("\n1. Prepare your data:")
        print("   - Ensure images are in: data/project-8-at-2026-01-07-07-09-0780865d/images/")
        print("   - Run: python scripts/preprocess_data.py")
        print("   - Run: python scripts/split_data.py")
    elif not model_ready:
        print("\n1. Train the model:")
        print("   - Run: python scripts/train.py --model yolov8s --epochs 100")
        print("   - This will take some time depending on your hardware")
    else:
        print("\n✓ Setup complete! You can now:")
        print("\n1. Evaluate the model:")
        print("   python scripts/evaluate.py --weights models/best.pt")
        print("\n2. Run inference on images/videos:")
        print("   python scripts/inference.py --source path/to/image.jpg --weights models/best.pt --show")
        print("\n3. Auto-label new images:")
        print("   python scripts/auto_label.py --source path/to/images --output path/to/labels --weights models/best.pt")
        print("\n4. Compare with separate models (if available):")
        print("   python scripts/compare_models.py --combined models/best.pt --ride models/ride.pt --cowtail models/cowtail.pt")

    print("\n" + "=" * 60)
    print("\nFor detailed instructions, see QUICKSTART.md")
    print("=" * 60)


def main():
    print("=" * 60)
    print("YOLO Combined Model - Setup Checker")
    print("=" * 60)

    # Run checks
    python_ok = check_python_version()
    deps_ok = check_dependencies()
    data_ready = check_data()
    model_ready = check_models()
    gpu_available = check_gpu()

    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    print(f"Python version: {'✓' if python_ok else '✗'}")
    print(f"Dependencies: {'✓' if deps_ok else '✗'}")
    print(f"Data ready: {'✓' if data_ready else '✗'}")
    print(f"Model trained: {'✓' if model_ready else '✗'}")
    print(f"GPU available: {'✓' if gpu_available else 'ℹ (CPU only)'}")

    # Suggest next steps
    suggest_next_steps(data_ready, model_ready)


if __name__ == "__main__":
    main()
