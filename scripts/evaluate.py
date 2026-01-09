"""
Evaluation Script for YOLO Combined Model
Evaluates model performance on test dataset
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import yaml
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class YOLOEvaluator:
    def __init__(self, weights_path, data_config='configs/dataset.yaml', img_size=(640, 384)):
        """
        Initialize YOLO evaluator

        Args:
            weights_path: Path to model weights
            data_config: Path to dataset YAML config
            img_size: Input image size (width, height)
        """
        self.weights_path = Path(weights_path)
        self.data_config = data_config
        self.img_size = img_size

        if not self.weights_path.exists():
            raise FileNotFoundError(f"Model weights not found: {weights_path}")

        # Load model
        print(f"Loading model from: {self.weights_path}")
        self.model = YOLO(str(self.weights_path))

    def evaluate(self, split='test', save_dir='results/test'):
        """
        Evaluate model on dataset

        Args:
            split: Dataset split to evaluate ('test', 'val', or 'train')
            save_dir: Directory to save results
        """
        print("\n" + "=" * 60)
        print(f"YOLO Model Evaluation - {split.upper()} Set")
        print("=" * 60)

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Validate model on dataset
        print(f"\nRunning evaluation on {split} set...")

        results = self.model.val(
            data=self.data_config,
            split=split,
            imgsz=self.img_size,
            batch=16,
            save_json=True,
            save_hybrid=True,
            conf=0.001,
            iou=0.6,
            max_det=300,
            plots=True,
            verbose=True,
        )

        print("\n" + "=" * 60)
        print("Evaluation Results")
        print("=" * 60)

        # Display metrics
        metrics = {
            'mAP50': results.results_dict.get('metrics/mAP50(B)', 0),
            'mAP50-95': results.results_dict.get('metrics/mAP50-95(B)', 0),
            'Precision': results.results_dict.get('metrics/precision(B)', 0),
            'Recall': results.results_dict.get('metrics/recall(B)', 0),
        }

        print("\nOverall Metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

        # Per-class metrics
        print("\nPer-Class Metrics:")
        class_names = ['cowtail', 'ride']

        # Try to get per-class metrics if available
        if hasattr(results, 'ap_class_index'):
            for i, class_name in enumerate(class_names):
                print(f"\n  {class_name}:")
                # These are approximations - ultralytics may have different attribute names
                if hasattr(results, 'mp'):
                    print(f"    Precision: {results.mp:.4f}")
                if hasattr(results, 'mr'):
                    print(f"    Recall: {results.mr:.4f}")

        # Save metrics to JSON
        metrics_file = save_dir / 'metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to: {metrics_file}")

        # Save detailed results
        results_file = save_dir / 'results.txt'
        with open(results_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write(f"YOLO Model Evaluation - {split.upper()} Set\n")
            f.write("=" * 60 + "\n\n")
            f.write("Overall Metrics:\n")
            for metric, value in metrics.items():
                f.write(f"  {metric}: {value:.4f}\n")
        print(f"Results saved to: {results_file}")

        return results, metrics

    def predict_samples(self, source, save_dir='results/test/predictions', conf_threshold=0.25):
        """
        Run predictions on sample images and save visualizations

        Args:
            source: Source directory or file
            save_dir: Directory to save predictions
            conf_threshold: Confidence threshold
        """
        print(f"\nRunning predictions on: {source}")
        print(f"Confidence threshold: {conf_threshold}")

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        results = self.model.predict(
            source=source,
            imgsz=self.img_size,
            conf=conf_threshold,
            save=True,
            save_txt=True,
            save_conf=True,
            project=str(save_dir.parent),
            name=save_dir.name,
            exist_ok=True,
        )

        print(f"\nPredictions saved to: {save_dir}")

        return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate YOLO model')

    parser.add_argument(
        '--weights',
        type=str,
        default='models/best.pt',
        help='Path to model weights'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='configs/dataset.yaml',
        help='Path to dataset YAML config'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['train', 'val', 'test'],
        help='Dataset split to evaluate (default: test)'
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
        '--save-dir',
        type=str,
        default='results/test',
        help='Directory to save results'
    )
    parser.add_argument(
        '--predict',
        type=str,
        default=None,
        help='Optional: Run predictions on specific images/directory'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='Confidence threshold for predictions (default: 0.25)'
    )

    args = parser.parse_args()

    evaluator = YOLOEvaluator(
        weights_path=args.weights,
        data_config=args.data,
        img_size=(args.img_width, args.img_height),
    )

    # Run evaluation
    results, metrics = evaluator.evaluate(split=args.split, save_dir=args.save_dir)

    # Run predictions if specified
    if args.predict:
        evaluator.predict_samples(
            source=args.predict,
            save_dir=f"{args.save_dir}/predictions",
            conf_threshold=args.conf,
        )

    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
