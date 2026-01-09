# YOLO Combined Model - Quick Start Guide

This guide will help you get started with training and using the combined ride and cowtail detection model.

## Prerequisites

- Python 3.8 or higher
- GPU recommended (but not required)
- Label Studio export data with ride and cowtail annotations

## Step 1: Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Step 2: Prepare Your Data

Place your Label Studio export data in the `data/` directory. The export folder can have any name (e.g., `data/my-project-export/`).

### Important: Locate Your Images

The Label Studio export contains labels but the images directory may be empty. You need to:

1. **Option A**: Download images from Label Studio
   - Export images along with annotations from Label Studio
   - Place them in `data/<your-export-folder>/images/`

2. **Option B**: Use existing images from another location
   - If you already have the images elsewhere, note that path
   - You'll provide it in the preprocessing step

### Preprocess the Data

Filter and remap the classes (keep only ride and cowtail):

```bash
# Specify your data folder with --input
python scripts/preprocess_data.py --input data/<your-export-folder>

# With images from a different location
python scripts/preprocess_data.py --input data/<your-export-folder> --images /path/to/your/images

# Include 10% background images (recommended to reduce false positives)
python scripts/preprocess_data.py --input data/<your-export-folder> --background-ratio 0.1

# Example with actual folder name
python scripts/preprocess_data.py --input data/project-8-at-2026-01-07-07-09-0780865d --background-ratio 0.1
```

**Preprocessing Options:**

| Option | Description |
|--------|-------------|
| `--input` | Path to Label Studio export folder (required) |
| `--output` | Output directory (default: `data/processed`) |
| `--images` | Path to images if not in export folder |
| `--background-ratio` | Ratio of background images to include (0.0-1.0, default: 0.0) |

This will:
- Filter labels to keep only "ride" and "cowtail"
- Remap class IDs: ride → 0, cowtail → 1
- Optionally add background images (images without target objects) to reduce false positives
- Clean up orphan labels (labels without matching images)
- Create processed dataset in `data/processed/`

**Combining Multiple Datasets:**

You can run the script multiple times with different input folders to stack data:

```bash
python scripts/preprocess_data.py --input data/dataset-A
python scripts/preprocess_data.py --input data/dataset-B
python scripts/preprocess_data.py --input data/dataset-C
```

> ⚠️ **Note on filename collisions**: If files have the same name across datasets, labels will be **overwritten** while images will be **skipped**. Ensure unique filenames across datasets, or clear `data/processed/` before each run if you don't want stacking.

### Split the Data

Split into train/validation/test sets:

```bash
python scripts/split_data.py
```

This creates:
- Training set: 70%
- Validation set: 20%
- Test set: 10%

### View Dataset with FiftyOne (Optional)

Visualize your dataset with FiftyOne to verify annotations before training.

> **Note**: FiftyOne requires Python 3.11 or lower. Use the separate `venv-fiftyone` environment.

```bash
# Activate FiftyOne environment
source venv-fiftyone/bin/activate

# View the processed dataset
python scripts/view_dataset.py --data data/processed

# View specific split only
python scripts/view_dataset.py --data data/processed --split val

# Use different port
python scripts/view_dataset.py --data data/processed --port 5151
```

Open http://localhost:5151 in your browser to explore images and annotations.

```bash
# When done, deactivate and switch back to main env
deactivate
source venv/bin/activate
```

## Step 3: Train the Model

Train YOLOv8s model (recommended):

```bash
python scripts/train.py --model yolov8s --epochs 100 --batch 16
```

**Training Options:**

```bash
# Use YOLOv11s instead
python scripts/train.py --model yolov11s

# Train longer with larger batch
python scripts/train.py --epochs 200 --batch 32

# Train on CPU only
python scripts/train.py --device cpu

# Resume interrupted training
python scripts/train.py --resume
```

**Expected Output:**
- Best model: `models/best.pt`
- Training results: `results/train/combined_model/`

## Step 4: Evaluate the Model

Test the model on test set:

```bash
python scripts/evaluate.py --weights models/best.pt
```

**Evaluation Metrics:**
- mAP50: Mean Average Precision at IoU=0.50
- mAP50-95: Mean Average Precision at IoU=0.50:0.95
- Precision and Recall

## Step 5: Compare with Separated Models (Optional)

If you have separate ride and cowtail models:

```bash
python scripts/compare_models.py \
  --combined models/best.pt \
  --ride models/ride_model.pt \
  --cowtail models/cowtail_model.pt
```

This will:
- Evaluate all models on the same test set
- Generate comparison plots
- Show improvement percentages
- Save results to `results/comparison/`

## Step 6: Run Inference

### On Images

```bash
# Single image
python scripts/inference.py --source path/to/image.jpg --weights models/best.pt --show

# Directory of images
python scripts/inference.py --source path/to/images/ --weights models/best.pt
```

### On Video

```bash
python scripts/inference.py --source video.mp4 --weights models/best.pt --show
```

### On Webcam/Stream

```bash
# Default webcam
python scripts/inference.py --source 0 --weights models/best.pt --show

# Specific camera
python scripts/inference.py --source 1 --weights models/best.pt --show
```

**Inference Options:**

```bash
# Adjust confidence threshold
python scripts/inference.py --source video.mp4 --conf 0.5

# Don't save output
python scripts/inference.py --source 0 --show --no-save

# Use different image size
python scripts/inference.py --source video.mp4 --img-width 1280 --img-height 768
```

## Step 7: Auto-Label New Images

Use the trained model to automatically label new unlabeled images:

```bash
python scripts/auto_label.py \
  --source path/to/new/images/ \
  --output path/to/output/ \
  --weights models/best.pt
```

**Auto-Labeling Options:**

```bash
# Save visualizations
python scripts/auto_label.py --source images/ --output labels/ --save-viz

# Copy images to output
python scripts/auto_label.py --source images/ --output labels/ --save-images

# Adjust confidence threshold
python scripts/auto_label.py --source images/ --output labels/ --conf 0.3
```

## Common Issues and Solutions

### Issue: No images found

**Solution**: Make sure to download/provide images before running preprocessing:
```bash
python scripts/preprocess_data.py --images /path/to/images
```

### Issue: CUDA out of memory

**Solution**: Reduce batch size:
```bash
python scripts/train.py --batch 8
```

### Issue: Low mAP

**Solutions**:
1. Train longer: `--epochs 200`
2. Use larger model: `--model yolov8m`
3. Check data quality and balance
4. Adjust confidence threshold during inference

### Issue: Slow inference on CPU

**Solution**: Use smaller model or enable GPU:
```bash
python scripts/train.py --model yolov8n  # Nano model (faster)
```

## Model Selection Guide

| Model | Speed | Accuracy | Size | Use Case |
|-------|-------|----------|------|----------|
| yolov8n | Fastest | Good | Smallest | Real-time on CPU |
| yolov8s | Fast | Better | Small | **Recommended** |
| yolov8m | Medium | Best | Medium | High accuracy needed |
| yolov8l | Slow | Excellent | Large | Offline processing |

## Tips for Best Results

1. **Data Quality**: Ensure annotations are accurate
2. **Balance**: Check class distribution (ride vs cowtail)
3. **Augmentation**: Training script includes data augmentation
4. **Validation**: Monitor validation metrics during training
5. **Fine-tuning**: Adjust confidence threshold for your use case

## Next Steps

1. Monitor training with TensorBoard:
   ```bash
   tensorboard --logdir results/train/combined_model
   ```

2. Export to ONNX for deployment:
   ```bash
   yolo export model=models/best.pt format=onnx
   ```

3. Optimize for inference:
   - Use FP16 precision on GPU
   - Use smaller input size if speed is critical
   - Consider model quantization

## Support

- Check `README.md` for full documentation
- Review training results in `results/train/`
- Check test results in `results/test/`
- Compare models in `results/comparison/`
