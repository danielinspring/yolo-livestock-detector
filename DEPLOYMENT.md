# Deployment Guide - GPU Environment (RTX 3090)

Complete guide for deploying this project to a GPU environment via GitHub.

## 🎯 Quick Start

```bash
# On GPU server
git clone <your-repo-url>
cd seung
./setup_gpu.sh
```

## 📋 Prerequisites on GPU Server

### System Requirements

- **GPU**: NVIDIA RTX 3090 (or compatible)
- **OS**: Ubuntu 20.04/22.04 (recommended) or Windows with WSL2
- **CUDA**: 11.8 or 12.1
- **cuDNN**: Compatible version with CUDA
- **Python**: 3.8 - 3.11 (3.10 recommended)
- **RAM**: 16GB+ recommended
- **Storage**: 50GB+ free space

### Check GPU and CUDA

```bash
# Check if GPU is detected
nvidia-smi

# Check CUDA version
nvcc --version

# Check CUDA availability in Python
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

## 🚀 Deployment Steps

### Step 1: Prepare Local Repository

On your local machine:

```bash
# Initialize git if not already done
git init

# Add all files (respects .gitignore)
git add .

# Commit
git commit -m "Initial commit: YOLO combined model project"

# Add remote (replace with your GitHub URL)
git remote add origin https://github.com/yourusername/your-repo.git

# Push to GitHub
git push -u origin main
```

### Step 2: Clone on GPU Server

```bash
# SSH into GPU server
ssh user@gpu-server

# Clone repository
git clone https://github.com/yourusername/your-repo.git
cd your-repo
```

### Step 3: Setup Environment

#### Option A: Automatic Setup (Recommended)

```bash
# Make setup script executable
chmod +x setup_gpu.sh

# Run setup
./setup_gpu.sh
```

#### Option B: Manual Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA support first
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install ultralytics opencv-python numpy pandas matplotlib seaborn pillow pyyaml tqdm scikit-learn streamlit plotly tensorboard

# Verify GPU is available
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

### Step 4: Upload Data

Since images/data are not in git (large files), you need to upload them separately:

#### Option A: SCP (from local machine)

```bash
# Upload images
scp -r data/your-images user@gpu-server:/path/to/repo/data/

# Or compressed
tar -czf data.tar.gz data/
scp data.tar.gz user@gpu-server:/path/to/repo/
# Then on server:
tar -xzf data.tar.gz
```

#### Option B: Download from Cloud Storage

```bash
# If data is on cloud storage
wget https://your-cloud-storage/dataset.zip
unzip dataset.zip -d data/
```

#### Option C: Use rsync (efficient for updates)

```bash
rsync -avz --progress data/ user@gpu-server:/path/to/repo/data/
```

### Step 5: Preprocess and Train

```bash
# Activate environment
source venv/bin/activate

# Preprocess data
python scripts/preprocess_data.py --images data/your-images

# Split dataset
python scripts/split_data.py

# Train on GPU (will automatically use GPU if available)
python scripts/train.py --model yolov8s --epochs 100 --batch 32 --device 0

# For multi-GPU training
python scripts/train.py --model yolov8s --epochs 100 --batch 64 --device 0,1
```

## 🔧 GPU-Specific Configuration

### Training Parameters for RTX 3090

The RTX 3090 has 24GB VRAM, allowing larger batch sizes:

```bash
# YOLOv8s (recommended)
python scripts/train.py --model yolov8s --batch 32 --epochs 100

# YOLOv8m (medium - more accurate)
python scripts/train.py --model yolov8m --batch 24 --epochs 100

# YOLOv8l (large - best accuracy)
python scripts/train.py --model yolov8l --batch 16 --epochs 100

# Mixed precision training (faster)
python scripts/train.py --model yolov8s --batch 32 --epochs 100 --device 0
```

### Monitor GPU Usage

```bash
# Watch GPU usage in real-time
watch -n 1 nvidia-smi

# Or install gpustat
pip install gpustat
gpustat -i 1
```

### TensorBoard Monitoring

```bash
# Start TensorBoard (in separate terminal)
source venv/bin/activate
tensorboard --logdir results/train/combined_model --bind_all

# Access from browser: http://gpu-server-ip:6006
```

## 📊 What NOT to Commit to Git

The `.gitignore` is configured to exclude:

✅ **Safe to commit:**
- Python scripts (`.py`)
- Configuration files (`.yaml`, `.md`)
- Requirements files
- Documentation

❌ **DO NOT commit:**
- Virtual environment (`venv/`)
- Dataset images/videos (`.jpg`, `.png`, `.mp4`)
- Trained models (`.pt`, `.pth`)
- Training results (`results/`)
- Processed data (`data/processed/`)
- Large files (>100MB)

## 🔄 Updating Code on GPU Server

When you update code locally:

```bash
# Local machine: Commit and push
git add .
git commit -m "Update training script"
git push

# GPU server: Pull updates
git pull origin main

# If dependencies changed
source venv/bin/activate
pip install -r requirements-gpu.txt
```

## 📦 Managing Large Files

### Option 1: Git LFS (for models)

If you want to version control trained models:

```bash
# Install Git LFS
git lfs install

# Track model files
git lfs track "models/*.pt"
git add .gitattributes
git commit -m "Configure Git LFS for models"

# Now you can commit models (up to size limit)
git add models/best.pt
git commit -m "Add trained model"
git push
```

### Option 2: Separate Storage

Keep large files in separate storage:
- **Cloud Storage**: AWS S3, Google Cloud Storage, Azure Blob
- **Shared Network**: NFS, Samba
- **Transfer Tools**: rsync, scp, rclone

### Option 3: DVC (Data Version Control)

For dataset versioning:

```bash
# Install DVC
pip install dvc dvc-s3  # or dvc-gs, dvc-azure

# Initialize DVC
dvc init

# Track data
dvc add data/processed
git add data/processed.dvc .dvc/config
git commit -m "Track data with DVC"

# Push data to remote storage
dvc remote add -d storage s3://mybucket/dvcstore
dvc push
```

## 🐛 Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size
python scripts/train.py --batch 16  # or 8

# Use gradient accumulation
# (edit train.py to add accumulation steps)
```

### GPU Not Detected

```bash
# Check NVIDIA driver
nvidia-smi

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Slow Training

```bash
# Enable mixed precision (AMP)
# Already enabled by default in train.py

# Check GPU utilization
nvidia-smi dmon -s u

# If GPU util is low, increase batch size or workers
python scripts/train.py --batch 32 --workers 8
```

### Permission Denied

```bash
# Fix script permissions
chmod +x setup_gpu.sh run_gui.sh

# Fix directory permissions
chmod -R 755 scripts/
```

## 🔐 Security Considerations

### SSH Keys

Use SSH keys instead of passwords:

```bash
# Generate SSH key (local)
ssh-keygen -t ed25519 -C "your_email@example.com"

# Copy to server
ssh-copy-id user@gpu-server
```

### Environment Variables

Don't commit secrets to git:

```bash
# Create .env file (already in .gitignore)
echo "API_KEY=your_secret_key" >> .env

# Load in Python
from dotenv import load_dotenv
load_dotenv()
```

## 📈 Performance Optimization

### RTX 3090 Optimal Settings

```bash
# Training
python scripts/train.py \
  --model yolov8s \
  --batch 32 \
  --epochs 100 \
  --img-width 640 \
  --img-height 384 \
  --device 0 \
  --workers 8

# Expected performance:
# - Training speed: ~150-200 images/sec
# - Epoch time: ~2-3 minutes (for 2000 images)
# - Total training: 3-5 hours (100 epochs)
```

### Monitoring Tools

```bash
# Install monitoring tools
pip install nvitop

# Run interactive monitor
nvitop

# Or use built-in
nvidia-smi dmon -s pucvmet -d 1
```

## 🌐 Running Web GUI on Server

### Local Access Only

```bash
source venv/bin/activate
streamlit run app.py
# Access: http://localhost:8501
```

### Remote Access

```bash
# Allow external connections
streamlit run app.py --server.address 0.0.0.0 --server.port 8501

# Access from browser: http://gpu-server-ip:8501
```

### SSH Tunnel (Secure)

```bash
# On local machine
ssh -L 8501:localhost:8501 user@gpu-server

# On server (in SSH session)
source venv/bin/activate
streamlit run app.py

# Access from local browser: http://localhost:8501
```

## 📝 Deployment Checklist

- [ ] GPU server has CUDA and cuDNN installed
- [ ] Python 3.8-3.11 available
- [ ] Git installed on server
- [ ] Repository pushed to GitHub
- [ ] SSH access configured
- [ ] Repository cloned on server
- [ ] Virtual environment created
- [ ] PyTorch with CUDA installed
- [ ] Dependencies installed
- [ ] GPU detected by PyTorch
- [ ] Data uploaded to server
- [ ] Data preprocessed and split
- [ ] Training started successfully
- [ ] Monitoring tools configured
- [ ] Backup strategy in place

## 🆘 Getting Help

- **PyTorch CUDA Issues**: https://pytorch.org/get-started/locally/
- **Ultralytics Docs**: https://docs.ultralytics.com
- **CUDA Toolkit**: https://developer.nvidia.com/cuda-toolkit
- **Project Issues**: Check logs in `results/train/` and `*.log` files

## 📚 Additional Resources

- **CUDA Installation**: https://docs.nvidia.com/cuda/
- **cuDNN Installation**: https://docs.nvidia.com/deeplearning/cudnn/
- **Docker Alternative**: Consider using NVIDIA Docker for containerized deployment
- **Jupyter Lab**: For interactive development on GPU server
