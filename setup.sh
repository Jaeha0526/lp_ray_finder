#!/bin/bash
# Setup script to install all necessary dependencies in virtual environment
# Run this ONCE before submitting SLURM jobs

echo "=========================================="
echo "Setting up LP Ray Finder environment"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if venv exists, create if not
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to create virtual environment${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}Virtual environment exists${NC}"
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate

# Upgrade pip first
echo -e "${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip setuptools wheel

# Install core dependencies
echo -e "${YELLOW}Installing core dependencies...${NC}"
pip install numpy scipy

# Check CUDA version and install appropriate JAX
echo -e "${YELLOW}Checking CUDA version...${NC}"
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//' | cut -d. -f1)
    echo "CUDA version detected: $CUDA_VERSION"
else
    # Try nvidia-smi
    if command -v nvidia-smi &> /dev/null; then
        CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | sed 's/.*CUDA Version: //' | cut -d. -f1)
        echo "CUDA version from nvidia-smi: $CUDA_VERSION"
    else
        echo -e "${YELLOW}CUDA not detected, will install CPU-only JAX${NC}"
        CUDA_VERSION=""
    fi
fi

# Install JAX with appropriate CUDA support
echo -e "${YELLOW}Installing JAX...${NC}"
if [ -n "$CUDA_VERSION" ]; then
    if [ "$CUDA_VERSION" -ge "12" ]; then
        echo "Installing JAX for CUDA 12.x"
        pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    elif [ "$CUDA_VERSION" -eq "11" ]; then
        echo "Installing JAX for CUDA 11.x"
        pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    else
        echo "Installing JAX for CUDA 11.x (fallback)"
        pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    fi
else
    echo "Installing CPU-only JAX"
    pip install --upgrade jax jaxlib
fi

# Install CuPy as fallback (optional)
echo -e "${YELLOW}Installing CuPy (optional, as fallback)...${NC}"
if [ -n "$CUDA_VERSION" ]; then
    if [ "$CUDA_VERSION" -ge "12" ]; then
        pip install cupy-cuda12x || echo -e "${YELLOW}CuPy installation failed (non-critical)${NC}"
    else
        pip install cupy-cuda11x || echo -e "${YELLOW}CuPy installation failed (non-critical)${NC}"
    fi
else
    echo "Skipping CuPy (requires CUDA)"
fi

# Install additional useful packages
echo -e "${YELLOW}Installing additional packages...${NC}"
pip install matplotlib tqdm

# Test installations
echo ""
echo -e "${YELLOW}Testing installations...${NC}"
echo "=========================================="

# Test numpy
python -c "import numpy; print(f'✓ NumPy {numpy.__version__}')" 2>/dev/null || echo -e "${RED}✗ NumPy import failed${NC}"

# Test scipy
python -c "import scipy; print(f'✓ SciPy {scipy.__version__}')" 2>/dev/null || echo -e "${RED}✗ SciPy import failed${NC}"

# Test JAX
python -c "import jax; print(f'✓ JAX {jax.__version__}'); print(f'  Devices: {jax.devices()}')" 2>/dev/null || echo -e "${RED}✗ JAX import failed${NC}"

# Test CuPy
python -c "import cupy; print(f'✓ CuPy {cupy.__version__}')" 2>/dev/null || echo -e "${YELLOW}✗ CuPy not available (optional)${NC}"

# Test matplotlib
python -c "import matplotlib; print(f'✓ Matplotlib {matplotlib.__version__}')" 2>/dev/null || echo -e "${YELLOW}✗ Matplotlib not available (optional)${NC}"

echo "=========================================="

# Create requirements.txt for reference
echo -e "${YELLOW}Creating requirements.txt for reference...${NC}"
pip freeze > requirements.txt

# Test if JAX can use GPU
echo ""
echo -e "${YELLOW}Testing GPU availability for JAX...${NC}"
python -c "
import jax
devices = jax.devices()
if any('gpu' in str(d).lower() for d in devices):
    print('✓ JAX GPU support is working!')
    print(f'  Available devices: {devices}')
else:
    print('⚠ JAX is using CPU only')
    print(f'  Available devices: {devices}')
" 2>/dev/null || echo -e "${RED}Could not test JAX GPU support${NC}"

echo ""
echo -e "${GREEN}=========================================="
echo -e "Setup complete!"
echo -e "==========================================${NC}"
echo ""
echo "To use this environment:"
echo "  1. Activate it: source venv/bin/activate"
echo "  2. Submit jobs: sbatch find_extremal_rays.sbatch"
echo ""
echo "The SLURM job will automatically use this pre-configured environment."