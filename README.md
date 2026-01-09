# YOLO Combined Object Detection Model

Combined YOLO model for detecting both **ride** and **cowtail** objects.

## Project Specifications

- **YOLO Version**: 8 or 11
- **Model Size**: YOLOv8s (small) for optimal performance
- **Input Size**: 640x384
- **Classes**:
  - 0: cowtail
  - 1: ride

## Project Structure

```
.
├── data/                          # Dataset directory
│   └── project-8-at-2026-01-07-07-09-0780865d/  # Label Studio export
├── scripts/                       # Python scripts
│   ├── preprocess_data.py        # Data preprocessing from Label Studio
│   ├── train.py                  # Training script
│   ├── evaluate.py               # Evaluation script
│   ├── compare_models.py         # Compare combined vs separated models
│   ├── inference.py              # Video/stream inference
│   └── auto_label.py             # Auto-labeling tool
├── configs/                       # Configuration files
│   └── dataset.yaml              # YOLO dataset configuration
├── models/                        # Trained models
├── results/                       # Output results
│   ├── train/                    # Training results
│   ├── test/                     # Test results
│   ├── comparison/               # Model comparison results
│   └── inference/                # Inference results
└── requirements.txt               # Python dependencies
```

## Setup

### Quick Setup with Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Mac/Linux
# or
venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

### Traditional Setup

```bash
pip install -r requirements.txt
```

## Web GUI

Launch the web interface for easy interaction with the model:

```bash
# Using the startup script
./run_gui.sh

# Or manually
source venv/bin/activate
streamlit run app.py
```

The GUI will open in your browser at http://localhost:8501

### GUI Features

- **📸 Inference**: Upload images/videos for real-time detection
- **🏷️ Auto-Label**: Batch label new images automatically
- **📊 Dataset Info**: View dataset statistics and visualizations

## Command Line Usage

### 1. Preprocess Data
```bash
# Basic preprocessing
python scripts/preprocess_data.py --input data/<your-export-folder>

# With 10% background images (recommended)
python scripts/preprocess_data.py --input data/<your-export-folder> --background-ratio 0.1
```

### 2. Train Model
```bash
python scripts/train.py
```

### 3. Evaluate Model
```bash
python scripts/evaluate.py --weights models/best.pt
```

### 4. Compare Models
```bash
python scripts/compare_models.py --combined models/best.pt --ride models/ride_model.pt --cowtail models/cowtail_model.pt
```

### 5. Run Inference
```bash
# Video file
python scripts/inference.py --source video.mp4 --weights models/best.pt

# Webcam/stream
python scripts/inference.py --source 0 --weights models/best.pt
```

### 6. Auto-Label
```bash
python scripts/auto_label.py --source images/ --weights models/best.pt --output labels/
```

## Notes

- Model trained on 640x384 input resolution
- Optimized for ride and cowtail detection
- Supports both image and video inference
