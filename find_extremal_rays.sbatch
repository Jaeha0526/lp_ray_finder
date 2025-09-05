#!/bin/bash
#SBATCH --job-name=n6_extremal_rays_jax
#SBATCH --output=logs/n6_extremal_jax_%j.out
#SBATCH --error=logs/n6_extremal_jax_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --partition=gpu

# Create logs directory if it doesn't exist
mkdir -p logs
mkdir -p results

# Load necessary modules (adjust based on your cluster)
# module load python/3.9
# module load cuda/11.8

# Activate virtual environment (REQUIRED - must be set up first with setup.sh)
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
    echo "Python: $(which python)"
    echo "Python version: $(python --version)"
else
    echo "ERROR: Virtual environment not found!"
    echo "Please run ./setup.sh first to install dependencies"
    exit 1
fi

# Set environment variables for GPU and JAX
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export XLA_PYTHON_CLIENT_PREALLOCATE=false  # Better memory management for JAX
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90  # Use 90% of GPU memory
export JAX_ENABLE_X64=false  # Use float32 for speed

# Print job information
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv

# Check installed packages (should already be installed via setup.sh)
echo "Checking installed packages..."
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import scipy; print(f'SciPy: {scipy.__version__}')"
python -c "import jax; print(f'JAX: {jax.__version__}')"

# Test JAX GPU availability
echo "Testing JAX GPU support..."
python -c "import jax; print(f'JAX devices: {jax.devices()}'); print(f'JAX version: {jax.__version__}')"

# Run benchmark first (optional, can comment out)
echo "Running JAX vs CuPy benchmark..."
python -c "from lp_ray_finder_jax import benchmark_jax_vs_cupy; benchmark_jax_vs_cupy()" 2>/dev/null || echo "Benchmark skipped"

# Run the ray finder with CORRECTED implementation
echo "Starting extremal ray search with CORRECTED constraint formulation..."
echo "Using homogeneous constraints (Ax >= 0, no constant term)"
python lp_ray_finder_correct.py \
    n6data/n6_correct_s7_expansion.ine \
    --job-id $SLURM_JOB_ID \
    --num-searches 1000

# Check exit status
if [ $? -eq 0 ]; then
    echo "Job completed successfully at: $(date)"
else
    echo "Job failed with error code: $?"
fi