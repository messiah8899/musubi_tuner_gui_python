#!/bin/bash

# Musubi Tuner v0.2.13 Training Interface - Linux Version
# Authors: suzuki & eddy
# Adapted for Linux compatibility

echo "========================================"
echo "  Musubi Tuner v0.2.13 Training GUI"
echo "  Authors: suzuki & eddy"
echo "  Linux Version"
echo "========================================"
echo

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to detect Python executable
detect_python() {
    if command_exists python3; then
        echo "python3"
    elif command_exists python; then
        # Check if it's Python 3
        if python -c "import sys; exit(0 if sys.version_info[0] == 3 else 1)" 2>/dev/null; then
            echo "python"
        else
            echo ""
        fi
    else
        echo ""
    fi
}

# Setup environment
echo "[INFO] Setting up environment for Linux..."

# Detect Python executable
PYTHON_CMD=$(detect_python)
if [ -z "$PYTHON_CMD" ]; then
    echo "[ERROR] Python 3 not found. Please install Python 3.10 or higher."
    exit 1
fi

echo "[OK] Python found: $PYTHON_CMD"

# Check Python version
PYTHON_VERSION=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "[INFO] Python version: $PYTHON_VERSION"

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "[INFO] Virtual environment found, activating..."
    source venv/bin/activate
    PYTHON_CMD="python"
elif [ -d ".venv" ]; then
    echo "[INFO] Virtual environment found, activating..."
    source .venv/bin/activate
    PYTHON_CMD="python"
else
    echo "[INFO] No virtual environment found, using system Python"
fi

# Check for CUDA availability
if command_exists nvidia-smi; then
    echo "[INFO] NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1
    
    # Check CUDA version
    if command_exists nvcc; then
        CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
        echo "[INFO] CUDA version: $CUDA_VERSION"
    else
        echo "[WARNING] CUDA compiler not found, but GPU detected"
    fi
else
    echo "[WARNING] No NVIDIA GPU detected, will use CPU"
fi

# Set environment variables for better performance
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

# Check if required packages are installed
echo "[INFO] Checking dependencies..."
$PYTHON_CMD -c "import torch; print(f'PyTorch version: {torch.__version__}')" 2>/dev/null || {
    echo "[ERROR] PyTorch not found. Please install requirements first:"
    echo "  pip install -r requirements.txt"
    exit 1
}

$PYTHON_CMD -c "import gradio; print(f'Gradio version: {gradio.__version__}')" 2>/dev/null || {
    echo "[ERROR] Gradio not found. Please install requirements first:"
    echo "  pip install -r requirements.txt"
    exit 1
}

echo "[INFO] Environment setup complete"
echo "[INFO] Available Attention implementations:"
echo "  - Flash-Attention (if installed)"
echo "  - Standard PyTorch Attention"
echo

echo "Starting Gradio interface..."
echo "Access URL: http://0.0.0.0:7860/ (Local)"
echo "External Access: http://YOUR_SERVER_IP:7860/"
echo "Press Ctrl+C to stop the server"
echo

# Start the training GUI
$PYTHON_CMD train_gui_new.py "$@"

# Check exit code
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo
    echo "[ERROR] Program exited with error code: $EXIT_CODE"
    echo "Please check the error messages above"
    echo
fi

echo "Press Enter to exit..."
read