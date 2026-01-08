"""
Model Comparison Script
Compares combined model with separate ride and cowtail models
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, Optional


class ModelComparator:
    def __init__(
        self,
        combined_model_path,
        ride_model_path=None,
        cowtail_model_path=None,
        data_config='configs/dataset.yaml',
        img_size=(640, 384),
    ):
        """
        Initialize model comparator

        Args:
            combined_model_path: Path to combined model weights
            ride_model_path: Path to separate ride model weights (optional)
            cowtail_model_path: Path to separate cowtail model weights (optional)
            data_config: Path to dataset YAML config
            img_size: Input image size (width, height)
        """
        self.data_config = data_config
        self.img_size = img_size
        self.results = {}

        # Load combined model
        print(f"Loading combined model: {combined_model_path}")
        if Path(combined_model_path).exists():
            self.combined_model = YOLO(combined_model_path)
            self.results['combined'] = {}
        else:
            raise FileNotFoundError(f"Combined model not found: {combined_model_path}")

        # Load separate models if provided
        self.ride_model = None
        self.cowtail_model = None

        if ride_model_path and Path(ride_model_path).exists():
            print(f"Loading ride model: {ride_model_path}")
            self.ride_model = YOLO(ride_model_path)
            self.results['ride'] = {}

        if cowtail_model_path and Path(cowtail_model_path).exists():
            print(f"Loading cowtail model: {cowtail_model_path}")
            self.cowtail_model = YOLO(cowtail_model_path)
            self.results['cowtail'] = {}

    def evaluate_combined_model(self, split='test'):
        """Evaluate combined model on test set"""
        print("\n" + "=" * 60)
        print("Evaluating Combined Model")
        print("=" * 60)

        results = self.combined_model.val(
            data=self.data_config,
            split=split,
            imgsz=self.img_size,
            batch=16,
            verbose=False,
        )

        metrics = {
            'mAP50': results.results_dict.get('metrics/mAP50(B)', 0),
            'mAP50-95': results.results_dict.get('metrics/mAP50-95(B)', 0),
            'Precision': results.results_dict.get('metrics/precision(B)', 0),
            'Recall': results.results_dict.get('metrics/recall(B)', 0),
            'F1': 2 * (results.results_dict.get('metrics/precision(B)', 0) *
                      results.results_dict.get('metrics/recall(B)', 0)) /
                      (results.results_dict.get('metrics/precision(B)', 0) +
                       results.results_dict.get('metrics/recall(B)', 0) + 1e-6),
        }

        self.results['combined'] = metrics

        print("\nCombined Model Metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

        return metrics

    def evaluate_separate_models(self, split='test'):
        """Evaluate separate ride and cowtail models"""
        print("\n" + "=" * 60)
        print("Evaluating Separate Models")
        print("=" * 60)

        # Evaluate ride model
        if self.ride_model:
            print("\nEvaluating Ride Model...")
            ride_results = self.ride_model.val(
                data=self.data_config,
                split=split,
                imgsz=self.img_size,
                batch=16,
                verbose=False,
            )

            ride_metrics = {
                'mAP50': ride_results.results_dict.get('metrics/mAP50(B)', 0),
                'mAP50-95': ride_results.results_dict.get('metrics/mAP50-95(B)', 0),
                'Precision': ride_results.results_dict.get('metrics/precision(B)', 0),
                'Recall': ride_results.results_dict.get('metrics/recall(B)', 0),
                'F1': 2 * (ride_results.results_dict.get('metrics/precision(B)', 0) *
                          ride_results.results_dict.get('metrics/recall(B)', 0)) /
                          (ride_results.results_dict.get('metrics/precision(B)', 0) +
                           ride_results.results_dict.get('metrics/recall(B)', 0) + 1e-6),
            }

            self.results['ride'] = ride_metrics

            print("\nRide Model Metrics:")
            for metric, value in ride_metrics.items():
                print(f"  {metric}: {value:.4f}")

        # Evaluate cowtail model
        if self.cowtail_model:
            print("\nEvaluating Cowtail Model...")
            cowtail_results = self.cowtail_model.val(
                data=self.data_config,
                split=split,
                imgsz=self.img_size,
                batch=16,
                verbose=False,
            )

            cowtail_metrics = {
                'mAP50': cowtail_results.results_dict.get('metrics/mAP50(B)', 0),
                'mAP50-95': cowtail_results.results_dict.get('metrics/mAP50-95(B)', 0),
                'Precision': cowtail_results.results_dict.get('metrics/precision(B)', 0),
                'Recall': cowtail_results.results_dict.get('metrics/recall(B)', 0),
                'F1': 2 * (cowtail_results.results_dict.get('metrics/precision(B)', 0) *
                          cowtail_results.results_dict.get('metrics/recall(B)', 0)) /
                          (cowtail_results.results_dict.get('metrics/precision(B)', 0) +
                           cowtail_results.results_dict.get('metrics/recall(B)', 0) + 1e-6),
            }

            self.results['cowtail'] = cowtail_metrics

            print("\nCowtail Model Metrics:")
            for metric, value in cowtail_metrics.items():
                print(f"  {metric}: {value:.4f}")

        # Calculate average of separate models
        if self.ride_model and self.cowtail_model:
            avg_metrics = {}
            for metric in ride_metrics.keys():
                avg_metrics[metric] = (ride_metrics[metric] + cowtail_metrics[metric]) / 2

            self.results['separate_avg'] = avg_metrics

            print("\nAverage of Separate Models:")
            for metric, value in avg_metrics.items():
                print(f"  {metric}: {value:.4f}")

    def compare_and_visualize(self, save_dir='results/comparison'):
        """Compare models and create visualizations"""
        print("\n" + "=" * 60)
        print("Model Comparison")
        print("=" * 60)

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Create comparison table
        df_data = []
        for model_name, metrics in self.results.items():
            row = {'Model': model_name}
            row.update(metrics)
            df_data.append(row)

        df = pd.DataFrame(df_data)

        # Save to CSV
        csv_path = save_dir / 'comparison.csv'
        df.to_csv(csv_path, index=False)
        print(f"\nComparison table saved to: {csv_path}")

        # Display comparison
        print("\nComparison Table:")
        print(df.to_string(index=False))

        # Calculate improvement
        if 'combined' in self.results and 'separate_avg' in self.results:
            print("\nImprovement of Combined Model over Separate Models:")
            for metric in self.results['combined'].keys():
                combined_val = self.results['combined'][metric]
                separate_val = self.results['separate_avg'][metric]
                improvement = ((combined_val - separate_val) / (separate_val + 1e-6)) * 100
                status = "✓" if improvement >= 0 else "✗"
                print(f"  {metric}: {improvement:+.2f}% {status}")

        # Create visualization
        self.plot_comparison(df, save_dir)

        # Save JSON
        json_path = save_dir / 'comparison.json'
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to: {json_path}")

    def plot_comparison(self, df, save_dir):
        """Create comparison plots"""
        metrics_to_plot = ['mAP50', 'mAP50-95', 'Precision', 'Recall', 'F1']

        # Set style
        sns.set_style("whitegrid")
        plt.figure(figsize=(12, 8))

        # Create bar plot
        x = np.arange(len(metrics_to_plot))
        width = 0.25

        models = df['Model'].tolist()
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']

        for i, model in enumerate(models):
            values = [df[df['Model'] == model][metric].values[0] for metric in metrics_to_plot]
            plt.bar(x + i * width, values, width, label=model, color=colors[i % len(colors)])

        plt.xlabel('Metrics', fontsize=12, fontweight='bold')
        plt.ylabel('Score', fontsize=12, fontweight='bold')
        plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
        plt.xticks(x + width * (len(models) - 1) / 2, metrics_to_plot)
        plt.legend()
        plt.ylim(0, 1.0)
        plt.tight_layout()

        plot_path = save_dir / 'comparison_plot.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {plot_path}")
        plt.close()

    def compare(self, split='test', save_dir='results/comparison'):
        """Run full comparison pipeline"""
        # Evaluate combined model
        self.evaluate_combined_model(split)

        # Evaluate separate models if available
        if self.ride_model or self.cowtail_model:
            self.evaluate_separate_models(split)

        # Compare and visualize
        self.compare_and_visualize(save_dir)

        print("\n" + "=" * 60)
        print("Comparison Complete!")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Compare YOLO models')

    parser.add_argument(
        '--combined',
        type=str,
        default='models/best.pt',
        help='Path to combined model weights'
    )
    parser.add_argument(
        '--ride',
        type=str,
        default=None,
        help='Path to ride model weights (optional)'
    )
    parser.add_argument(
        '--cowtail',
        type=str,
        default=None,
        help='Path to cowtail model weights (optional)'
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
        default='results/comparison',
        help='Directory to save results'
    )

    args = parser.parse_args()

    comparator = ModelComparator(
        combined_model_path=args.combined,
        ride_model_path=args.ride,
        cowtail_model_path=args.cowtail,
        data_config=args.data,
        img_size=(args.img_width, args.img_height),
    )

    comparator.compare(split=args.split, save_dir=args.save_dir)


if __name__ == "__main__":
    main()
