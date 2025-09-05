#!/bin/bash
#SBATCH --job-name=n6_10k_rays
#SBATCH --output=logs/n6_10k_rays_%j.out
#SBATCH --error=logs/n6_10k_rays_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
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
echo "============================================================"
echo "N=6 EXTREME RAY FINDER - 10,000 RAY SEARCH"
echo "============================================================"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Target: 10,000 extreme rays"
echo "Estimated time: ~85 hours (3.5 days)"
echo ""
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

# Check installed packages (should already be installed via setup.sh)
echo "Checking installed packages..."
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import scipy; print(f'SciPy: {scipy.__version__}')"
python -c "import jax; print(f'JAX: {jax.__version__}')" 2>/dev/null || echo "JAX: Not available (CPU only)"

# Test JAX GPU availability (optional)
echo ""
echo "Testing JAX GPU support..."
python -c "import jax; print(f'JAX devices: {jax.devices()}'); print(f'JAX version: {jax.__version__}')" 2>/dev/null || echo "JAX GPU test skipped"

echo ""
echo "============================================================"
echo "STARTING 10,000 RAY SEARCH"
echo "============================================================"

# Run the ray finder with CORRECTED implementation for 10,000 rays
python lp_ray_finder_correct.py \
    n6data/n6_correct_s7_expansion.ine \
    --job-id $SLURM_JOB_ID \
    --num-searches 10000

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "JOB COMPLETED SUCCESSFULLY!"
    echo "============================================================"
    echo "Job completed at: $(date)"
    echo "Found 10,000 extreme rays of the N=6 holographic entropy cone"
    echo "Results saved in: results/job_${SLURM_JOB_ID}_*/"
    
    # Quick summary
    if [ -d "results/job_${SLURM_JOB_ID}_"* ]; then
        echo ""
        echo "Quick verification:"
        RESULT_DIR=$(ls -d results/job_${SLURM_JOB_ID}_* | head -1)
        if [ -d "$RESULT_DIR/rays" ]; then
            RAY_COUNT=$(ls "$RESULT_DIR/rays/"*.npz 2>/dev/null | wc -l)
            echo "  Ray files found: $RAY_COUNT"
        fi
        if [ -f "$RESULT_DIR/summary.json" ]; then
            echo "  Summary file: Found"
        fi
    fi
    
else
    echo ""
    echo "============================================================"
    echo "JOB FAILED!"
    echo "============================================================"
    echo "Job failed with error code: $?"
    echo "Check logs for details: logs/n6_10k_rays_${SLURM_JOB_ID}.err"
fi