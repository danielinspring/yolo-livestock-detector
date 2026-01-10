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

# Install albumentations for data augmentation (recommended)
pip install albumentations
```

## Step 1.5: Password Setup (Optional)

The Streamlit app requires password authentication. The default password is `admin123`.

To change the password:

```bash
# Create or edit the secrets file
echo 'password = "your_new_password"' > .streamlit/secrets.toml
```

> ⚠️ **Note**: The `.streamlit/secrets.toml` file is gitignored for security. Never commit passwords to version control.

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

# Disable automatic augmentation
python scripts/preprocess_data.py --input data/<your-export-folder> --no-augment

# Custom target ratio for augmentation (default is 2:1)
python scripts/preprocess_data.py --input data/<your-export-folder> --target-ratio 1.5

# Example with actual folder name
python scripts/preprocess_data.py --input data/project-8-at-2026-01-07-07-09-0780865d
```

**Preprocessing Options:**

| Option | Description |
|--------|-------------|
| `--input` | Path to Label Studio export folder (required) |
| `--output` | Output directory (default: `data/dataset/processed_YYYYMMDD_HHMMSS`) |
| `--images` | Path to images if not in export folder |
| `--background-ratio` | Ratio of background images to include (default: 0.065 = 6.5%) |
| `--no-augment` | Disable automatic minority class augmentation |
| `--target-ratio` | Target ride:cowtail ratio for augmentation (default: 2.0 for 2:1) |

This will:
- Filter labels to keep only "ride" and "cowtail"
- Remap class IDs: cowtail → 0, ride → 1
- Add background images (6.5% by default) to reduce false positives
- **Auto-augment minority class (cowtail)** when imbalance exceeds 2.5:1 ratio, targeting 2:1 using:
  - HorizontalFlip (좌우 반전)
  - RandomBrightness (밝기 조절)
  - Rotate ±15° (회전)
  - GaussNoise (노이즈)
  - *Note: Augmentation only triggers if ride:cowtail ratio > 2.5:1*
- Clean up orphan labels (labels without matching images)
- Create timestamped dataset in `data/dataset/processed_YYYYMMDD_HHMMSS/`

**Combining Multiple Datasets:**

You can run the script multiple times with different input folders to the same output:

```bash
# Specify the same output directory to combine datasets
python scripts/preprocess_data.py --input data/dataset-A --output data/dataset/combined
python scripts/preprocess_data.py --input data/dataset-B --output data/dataset/combined
python scripts/preprocess_data.py --input data/dataset-C --output data/dataset/combined
```

> ⚠️ **Note on filename collisions**: If files have the same name across datasets, labels will be **overwritten** while images will be **skipped**. Ensure unique filenames across datasets.

### Split the Data

Split into train/validation/test sets:

```bash
# Split a specific dataset (use the timestamped folder name)
python scripts/split_data.py --data-dir data/dataset/processed_20260110_143022

# Custom split ratios
python scripts/split_data.py --data-dir data/dataset/processed_20260110_143022 --train-ratio 0.8 --val-ratio 0.15 --test-ratio 0.05
```

**Split Options:**

| Option | Description |
|--------|-------------|
| `--data-dir` | Path to processed dataset folder |
| `--train-ratio` | Training set ratio (default: 0.7) |
| `--val-ratio` | Validation set ratio (default: 0.2) |
| `--test-ratio` | Test set ratio (default: 0.1) |
| `--seed` | Random seed for reproducibility (default: 42) |

This creates:
- Training set: 70% (default)
- Validation set: 20% (default)
- Test set: 10% (default)

### View Dataset with FiftyOne (Optional)

Visualize your dataset with FiftyOne to verify annotations before training.

> **Note**: FiftyOne requires Python 3.11 or lower. Use the separate `venv-fiftyone` environment.

```bash
# Activate FiftyOne environment
source venv-fiftyone/bin/activate

# View the processed dataset (use your timestamped folder)
python scripts/view_dataset.py --data data/dataset/processed_20260110_143022

# View specific split only
python scripts/view_dataset.py --data data/dataset/processed_20260110_143022 --split val

# Use different port
python scripts/view_dataset.py --data data/dataset/processed_20260110_143022 --port 5151
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
2. **Balance**: Preprocessing auto-augments minority class to 2:1 ratio (when > 2.5:1)
3. **Augmentation**: Both preprocessing (class balance) and training include augmentation
4. **Validation**: Monitor validation metrics during training
5. **Fine-tuning**: Adjust confidence threshold for your use case
6. **Streamlit UI**: Use `streamlit run Home.py` for visual dataset management
   - **Dataset Info** page: Select datasets, view statistics, split datasets with custom ratios
   - **Evaluation** page: Select models, run evaluation, compare metrics

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
