# GitHub Deployment Quick Reference

Fast reference guide for deploying to GPU server via GitHub.

## 📤 Push to GitHub (Local Machine)

```bash
# First time setup
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main

# Subsequent updates
git add .
git commit -m "Your commit message"
git push
```

## 📥 Deploy to GPU Server

```bash
# SSH into GPU server
ssh user@gpu-server

# Clone repository (first time only)
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO

# Run automated setup
./setup_gpu.sh

# Upload data (from local machine)
scp -r data/images user@gpu-server:/path/to/YOUR_REPO/data/

# Back on server: Preprocess and train
source venv/bin/activate
python scripts/preprocess_data.py --images data/images
python scripts/split_data.py
python scripts/train.py --model yolov8s --epochs 100 --batch 32
```

## 🔄 Update Code on Server

```bash
# On GPU server
cd YOUR_REPO
git pull
source venv/bin/activate
# If requirements changed:
pip install -r requirements-gpu.txt
```

## ⚠️ Important Notes

### What's Committed to Git
✅ **YES - Commit these:**
- Python scripts (`.py`)
- Config files (`.yaml`, `.md`, `.txt`)
- Documentation
- Small test files

❌ **NO - Don't commit these:**
- Virtual environment (`venv/`)
- Dataset images (`.jpg`, `.png`)
- Trained models (`.pt`)
- Results (`results/`)
- Large files (>50MB)

### File Sizes
- GitHub limit: **100 MB per file**
- Total repo: Keep under **1 GB** (soft limit)
- Use Git LFS for files 50MB-100MB
- Use separate storage for larger files

### Data Transfer Options

**Option 1: SCP (Simple)**
```bash
scp -r data/images user@gpu-server:/path/to/repo/data/
```

**Option 2: Rsync (Efficient)**
```bash
rsync -avz --progress data/ user@gpu-server:/path/to/repo/data/
```

**Option 3: Cloud Storage**
```bash
# Upload to S3/GCS/Azure
aws s3 cp data/ s3://your-bucket/data/ --recursive

# Download on server
aws s3 sync s3://your-bucket/data/ /path/to/repo/data/
```

## 🎯 RTX 3090 Optimal Settings

```bash
# Small model - Fast training
python scripts/train.py --model yolov8s --batch 32 --epochs 100

# Medium model - Balanced
python scripts/train.py --model yolov8m --batch 24 --epochs 100

# Large model - Best accuracy
python scripts/train.py --model yolov8l --batch 16 --epochs 100
```

## 📊 Monitor Training

```bash
# Terminal 1: Training
python scripts/train.py --model yolov8s --batch 32

# Terminal 2: TensorBoard
tensorboard --logdir results/train --bind_all

# Terminal 3: GPU monitor
watch -n 1 nvidia-smi
```

## 🐛 Common Issues

### "CUDA out of memory"
```bash
# Reduce batch size
python scripts/train.py --batch 16  # or 8
```

### "GPU not found"
```bash
# Check GPU
nvidia-smi

# Reinstall PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### "Permission denied" for scripts
```bash
chmod +x setup_gpu.sh run_gui.sh
```

## 🚀 Quick Commands Cheat Sheet

```bash
# Setup
./setup_gpu.sh

# Preprocess
python scripts/preprocess_data.py --images data/images

# Train
python scripts/train.py --model yolov8s --batch 32 --epochs 100

# Evaluate
python scripts/evaluate.py --weights models/best.pt

# Inference
python scripts/inference.py --source image.jpg --weights models/best.pt

# GUI
./run_gui.sh

# Monitor GPU
nvidia-smi
watch -n 1 nvidia-smi
```

## 📚 Full Documentation

- **Detailed Deployment**: `DEPLOYMENT.md`
- **Quick Start**: `QUICKSTART.md`
- **GUI Guide**: `GUI_GUIDE.md`
- **Main README**: `README.md`
