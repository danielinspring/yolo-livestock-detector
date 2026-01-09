# Web GUI Guide

Complete guide for using the YOLO Combined Model web interface.

## Starting the GUI

### Method 1: Using the Startup Script (Recommended)

```bash
./run_gui.sh
```

### Method 2: Manual Start

```bash
source venv/bin/activate
streamlit run app.py
```

The GUI will automatically open in your browser at **http://localhost:8501**

## Features Overview

### 🏠 Home Page

The main dashboard showing:
- Quick navigation to all features
- Model status (trained/not trained)
- Dataset status (ready/not ready)
- Quick start guide
- Configuration info

### 📸 Inference Page

**Upload and detect objects in images or videos**

#### Image Inference

1. Click "Browse files" to upload images (JPG, PNG, BMP)
2. Adjust settings in sidebar:
   - **Confidence Threshold**: Lower = more detections, Higher = fewer but more certain
   - **IoU Threshold**: Controls overlapping box merging
3. Click "🚀 Run Detection"
4. View results:
   - Side-by-side original and detected images
   - Detection counts (total, ride, cowtail)
   - Inference time
   - Detection details
5. Download result images

**Tips:**
- Upload multiple images at once for batch processing
- Hover over metrics to see explanations
- Expand "Detection Details" to see confidence scores

#### Video Inference

1. Upload video file (MP4, AVI, MOV, MKV)
2. View video info (FPS, frames, resolution)
3. Adjust "Process every N frames" slider:
   - **1**: Process every frame (slower, most accurate)
   - **5-10**: Skip frames (faster, good for long videos)
   - **30**: Very fast but might miss detections
4. Click "🚀 Process Video"
5. Watch real-time progress
6. Download processed video with detection boxes

**Tips:**
- For long videos, use higher N values
- Processing shows current frame every 10 frames
- Total detection count shown after processing

### 🏷️ Auto-Label Page

**Automatically generate YOLO labels for new images**

#### How to Use

1. **Upload Images**: Select multiple images to label
2. **Configure Settings**:
   - **Confidence Threshold**: Minimum confidence for labels (0.25 recommended)
   - **Output Format**:
     - YOLO TXT: Standard YOLO format
     - JSON: Detailed format with metadata
     - Both: Get both formats
   - **Include Images**: Copy images to download package
   - **Show Visualizations**: Preview detections
3. **Generate**: Click "🚀 Generate Labels"
4. **Review**: Expand preview to check labeling quality
5. **Download**: Get ZIP package with:
   - `labels/` - Label files
   - `images/` - Original images (if enabled)
   - `stats.json` - Labeling statistics
   - `README.md` - Package info

#### Output Formats

**YOLO TXT Format** (example.txt):
```
0 0.456789 0.523456 0.234567 0.345678
1 0.678901 0.234567 0.123456 0.234567
```
Format: `<class_id> <x_center> <y_center> <width> <height>` (all normalized 0-1)

**JSON Format** (example.json):
```json
{
  "image": "example.jpg",
  "width": 640,
  "height": 384,
  "annotations": [
    {
      "class_id": 0,
      "class_name": "ride",
      "confidence": 0.95,
      "bbox": {
        "x_center": 0.456,
        "y_center": 0.523,
        "width": 0.234,
        "height": 0.345
      },
      "bbox_pixels": {
        "x1": 100,
        "y1": 150,
        "x2": 250,
        "y2": 280
      }
    }
  ]
}
```

**Tips:**
- Review visualizations before downloading
- Lower confidence = more labels but more false positives
- Use auto-labeled data for quick initial labeling, then manually review
- Re-label after training better models

### 📊 Dataset Info Page

**View comprehensive dataset statistics and health checks**

#### Statistics Shown

1. **Overview Metrics**:
   - Total images
   - Total annotations
   - Ride/cowtail counts

2. **Dataset Splits**:
   - Train/Val/Test distribution
   - Images and annotations per split
   - Average annotations per image

3. **Visualizations**:
   - Images distribution pie chart
   - Annotations distribution pie chart
   - Class distribution bar chart
   - Overall class balance

4. **Dataset Health Check**:
   - ✅ Sufficient images check
   - ✅ Class balance check
   - ✅ Training set size check
   - ✅ Annotation density check

5. **Recommendations**:
   - Automatic suggestions for improving dataset quality
   - Warnings about potential issues

#### Export Statistics

Click "📥 Download Statistics as JSON" to get:
- Complete dataset statistics
- Class distribution per split
- All metrics in machine-readable format

**Tips:**
- Check health status before training
- Aim for balanced classes (50/50 ratio)
- Ensure train set is at least 60% of total
- Look for recommendations to improve dataset

## Settings & Configuration

### Sidebar Settings (Available on all pages)

**Model Path**: Path to trained model weights
- Default: `models/best.pt`
- Change if using different model

**Confidence Threshold**: Minimum confidence for detections
- Range: 0.0 - 1.0
- Default: 0.25
- Higher = fewer but more certain detections
- Lower = more detections but may include false positives

**IoU Threshold**: Intersection over Union for NMS
- Range: 0.0 - 1.0
- Default: 0.45
- Controls how overlapping boxes are merged
- Higher = allows more overlap
- Lower = more aggressive merging

### Class Information

The model detects two classes:
- **Class 0**: cowtail (displayed in green)
- **Class 1**: ride (displayed in blue/red)

### Input Size

Model is trained on: **640 x 384** pixels
- Images are automatically resized for inference
- Original aspect ratio preserved

## Troubleshooting

### Model Not Found

**Error**: "❌ Model not found: models/best.pt"

**Solution**:
1. Train a model first: `python scripts/train.py`
2. Or specify different model path in sidebar

### Dataset Not Ready

**Warning**: "⚠️ No processed dataset"

**Solution**:
1. Run preprocessing: `python scripts/preprocess_data.py`
2. Split dataset: `python scripts/split_data.py`

### Slow Inference

**Issue**: Detection takes too long

**Solution**:
- For video: Increase "Process every N frames"
- Close other applications
- Use smaller images
- Consider using GPU (if available)

### No Detections

**Issue**: Model doesn't detect anything

**Solution**:
- Lower confidence threshold
- Check if objects are clearly visible
- Verify model is trained properly
- Try different images

### Browser Won't Open

**Issue**: GUI doesn't open in browser

**Solution**:
1. Manually open: http://localhost:8501
2. Check if port is already in use
3. Try different port: `streamlit run app.py --server.port 8502`

## Keyboard Shortcuts

- **r**: Rerun the app
- **c**: Clear cache
- **Ctrl+C**: Stop the server (in terminal)

## Performance Tips

1. **Batch Processing**: Upload multiple images at once
2. **Video Optimization**: Use frame skipping for long videos
3. **Cache**: Streamlit caches model loading for faster subsequent runs
4. **Browser**: Use Chrome/Firefox for best performance
5. **Close Tabs**: Keep only one GUI tab open

## Advanced Usage

### Running on Different Port

```bash
streamlit run app.py --server.port 8502
```

### Running on Network

To access from other devices on your network:

```bash
streamlit run app.py --server.address 0.0.0.0
```

Then access from other devices: `http://<your-ip>:8501`

### Custom Theme

Create `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#2ecc71"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"
```

## Support & Resources

- **Quick Start**: See `QUICKSTART.md`
- **Full Documentation**: See `README.md`
- **Command Line**: All features available via CLI scripts
- **YOLO Docs**: https://docs.ultralytics.com

## Tips for Best Results

1. **Start Simple**: Test with a few images first
2. **Tune Thresholds**: Adjust confidence based on your needs
3. **Check Dataset**: Use Dataset Info page before training
4. **Review Labels**: Always review auto-labeled data
5. **Monitor Training**: Use TensorBoard for detailed metrics
6. **Compare Models**: Use comparison features to validate improvements

Enjoy using the YOLO Combined Model! 🎯
