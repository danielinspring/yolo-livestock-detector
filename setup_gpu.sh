#!/bin/bash

# YOLO Combined Model - GPU Environment Setup Script
# For RTX 3090 and other NVIDIA GPUs with CUDA support

set -e  # Exit on error

echo "=========================================="
echo "YOLO Combined Model - GPU Setup"
echo "=========================================="
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running on Linux/Unix
if [[ "$OSTYPE" != "linux-gnu"* ]] && [[ "$OSTYPE" != "darwin"* ]]; then
    echo -e "${RED}Warning: This script is designed for Linux/Unix systems${NC}"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check Python version
echo "Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python 3 is not installed!${NC}"
    echo "Please install Python 3.8-3.11"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo -e "${GREEN}✓ Python $PYTHON_VERSION detected${NC}"

# Check for NVIDIA GPU
echo ""
echo "Checking for NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓ NVIDIA GPU detected:${NC}"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

    # Check CUDA
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
        echo -e "${GREEN}✓ CUDA version: $CUDA_VERSION${NC}"
    else
        echo -e "${YELLOW}⚠ CUDA compiler (nvcc) not found in PATH${NC}"
        echo "  CUDA might still be available to PyTorch"
    fi
else
    echo -e "${YELLOW}⚠ nvidia-smi not found - GPU may not be available${NC}"
    read -p "Continue with CPU-only setup? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ -d "venv" ]; then
    echo -e "${YELLOW}Virtual environment already exists${NC}"
    read -p "Remove and recreate? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf venv
        python3 -m venv venv
        echo -e "${GREEN}✓ Virtual environment recreated${NC}"
    fi
else
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel
echo -e "${GREEN}✓ Pip upgraded${NC}"

# Detect CUDA version for PyTorch
echo ""
echo "Detecting CUDA version for PyTorch installation..."

CUDA_AVAILABLE=false
CUDA_VERSION_SHORT=""

if command -v nvidia-smi &> /dev/null; then
    # Try to get CUDA version from nvidia-smi
    CUDA_VERSION_FULL=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')

    if [ ! -z "$CUDA_VERSION_FULL" ]; then
        CUDA_AVAILABLE=true
        # Extract major.minor version (e.g., 11.8, 12.1)
        CUDA_VERSION_SHORT=$(echo $CUDA_VERSION_FULL | cut -d'.' -f1,2)
        echo -e "${GREEN}CUDA Version detected: $CUDA_VERSION_SHORT${NC}"
    fi
fi

# Install PyTorch with appropriate CUDA version
echo ""
echo "Installing PyTorch..."

if [ "$CUDA_AVAILABLE" = true ]; then
    # Map CUDA version to PyTorch installation command
    if [[ "$CUDA_VERSION_SHORT" == "12."* ]]; then
        echo "Installing PyTorch with CUDA 12.1 support..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    elif [[ "$CUDA_VERSION_SHORT" == "11.8"* ]]; then
        echo "Installing PyTorch with CUDA 11.8 support..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        echo -e "${YELLOW}CUDA version $CUDA_VERSION_SHORT detected${NC}"
        echo "Supported versions are 11.8 and 12.x"
        echo ""
        echo "Choose PyTorch installation:"
        echo "1) CUDA 11.8"
        echo "2) CUDA 12.1"
        echo "3) CPU only"
        read -p "Enter choice (1-3): " cuda_choice

        case $cuda_choice in
            1)
                pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
                ;;
            2)
                pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
                ;;
            3)
                pip install torch torchvision torchaudio
                ;;
            *)
                echo -e "${RED}Invalid choice${NC}"
                exit 1
                ;;
        esac
    fi
else
    echo "No CUDA detected, installing CPU-only PyTorch..."
    pip install torch torchvision torchaudio
fi

echo -e "${GREEN}✓ PyTorch installed${NC}"

# Verify PyTorch CUDA
echo ""
echo "Verifying PyTorch installation..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}' if torch.cuda.is_available() else 'CPU only'); print(f'GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else '')"

# Install other dependencies
echo ""
echo "Installing project dependencies..."
pip install ultralytics opencv-python numpy pandas matplotlib seaborn pillow pyyaml tqdm scikit-learn streamlit plotly tensorboard

echo -e "${GREEN}✓ All dependencies installed${NC}"

# Create necessary directories
echo ""
echo "Creating project directories..."
mkdir -p models results data/processed
touch models/.gitkeep results/.gitkeep

echo -e "${GREEN}✓ Directories created${NC}"

# Verify installation
echo ""
echo "=========================================="
echo "Verifying Installation"
echo "=========================================="
echo ""

echo "Python packages:"
pip list | grep -E "torch|ultralytics|streamlit|opencv"

echo ""
echo "GPU Status:"
python3 -c "
import torch
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU Device: {torch.cuda.get_device_name(0)}')
    print(f'  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')
    print(f'  CUDA Version: {torch.version.cuda}')
else:
    print('  Running on CPU')
"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Upload your dataset:"
echo "   scp -r data/images user@server:/path/to/repo/data/"
echo ""
echo "2. Preprocess data:"
echo "   source venv/bin/activate"
echo "   python scripts/preprocess_data.py --images data/images"
echo ""
echo "3. Split dataset:"
echo "   python scripts/split_data.py"
echo ""
echo "4. Train model:"
echo "   python scripts/train.py --model yolov8s --epochs 100 --batch 32"
echo ""
echo "5. Launch web GUI:"
echo "   ./run_gui.sh"
echo ""
echo "For detailed instructions, see DEPLOYMENT.md"
echo ""
