# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an LP-based extreme ray finder for convex cones, specifically designed for the N=6 holographic entropy cone system in quantum information theory. 

**IMPORTANT UPDATE**: The previously reported 129 "new" rays were found to NOT be true extreme rays due to lacking proper extremality verification. The algorithm has since been enhanced with rank-based extremality checking to ensure all discovered rays are genuinely extreme (lying on codimension-1 faces).

## Key Commands

### Environment Setup
```bash
# Activate virtual environment (if exists)
source venv/bin/activate

# Install core dependencies
pip install numpy scipy

# Install GPU acceleration (strongly recommended)
pip install cupy-cuda11x  # or cupy-cuda12x based on CUDA version

# Check CUDA version
nvcc --version
```

### Running the Ray Finder
```bash
# Phase 1: Basic implementation (testing only)
python lp_ray_finder.py

# Phase 2: CPU-optimized with multiprocessing
python lp_ray_finder_phase2.py

# Phase 3: GPU-accelerated (production - recommended)
python lp_ray_finder_phase3.py

# Find new rays with full S7 constraints (8.6M constraints)
python phase3_no_limit.py

# Extended search for multiple rays
python find_new_n6_rays.py
```

### Analysis and Verification
```bash
# Convert rays to integer coordinates
python convert_to_integer_rays.py

# Extract integer coordinates from existing rays
python extract_integer_coordinates.py

# Check if discovered rays are truly new
python check_against_known_rays.py
```

## Architecture

### Core Algorithm Implementation
The system uses an **active-set Linear Programming approach with extremality verification**:

1. **Active-Set LP Method**: Starts with small subset (~500-1000) of 8.6M constraints, iteratively adds violated constraints until convergence
2. **Extremality Verification** (CRITICAL - added after initial implementation): Ensures rays lie on codimension-1 faces by checking rank of tight constraints equals d-1. Without this, the algorithm may find points that satisfy constraints but are NOT extreme rays.
3. **Three-Phase Architecture**:
   - `lp_ray_finder.py`: Basic scipy implementation
   - `lp_ray_finder_phase2.py`: CPU multiprocessing optimization
   - `lp_ray_finder_phase3.py`: GPU-accelerated with CuPy (100-1000x speedup)

### Key Classes and Methods
- **`HybridLPRayFinder`** (phase3): Main production class with GPU support
  - `find_extreme_ray()`: Core algorithm WITH extremality verification (lines 386-434)
  - `GPUConstraintManager`: Handles constraint violation checking on GPU
- **`FullConstraintRayFinder`** (phase3_no_limit.py): Handles full 8.6M constraints
  - `find_extreme_ray_active_set()`: May need extremality verification added

### Data Files
- **Input**: `n6data/n6_correct_s7_expansion.ine` - 8.6M constraints defining N=6 cone
- **Results**: 
  - `truly_new_rays_final.txt` - **NOTE: These were found to NOT be true extreme rays**
  - `all_unique_rays_integer.txt` - Integer coordinate representations (not verified as extreme)
  - Various orbit representative files for S7 symmetry analysis

## Critical Context

### N=6 Holographic Entropy Cone
- **Dimensions**: 63
- **Constraints**: 8,665,853 (full S7 expansion)
- **S7 symmetry group**: 5,040 permutations
- **Computational challenge**: Traditional vertex enumeration would take decades-millennia

### Performance Characteristics
- **GPU Required**: CPU-only is 100x slower
- **Memory**: 15GB+ RAM needed for full constraint set
- **Per ray**: 30 seconds to 5 minutes with GPU
- **Critical**: Without extremality verification, results may not be true extreme rays

### Algorithm Guarantees (After Extremality Fix)
- All rays satisfy the full 8.6M constraint system
- **Extremality now verified through rank checking** (ensures rank = d-1 for d dimensions)
- Adaptive constraint addition prevents rank-deficient solutions
- **Previous results without this verification were NOT true extreme rays**

## Important Patterns

### Finding New Rays
```python
from lp_ray_finder_phase3 import HybridLPRayFinder
import numpy as np

finder = HybridLPRayFinder(
    'n6data/n6_correct_s7_expansion.ine',
    verbose=True,
    use_gpu=True,
    chunk_size=500000
)

# Random objective for diversity
objective = np.random.randn(finder.d)
objective = objective / np.linalg.norm(objective)

ray, stats = finder.find_extreme_ray(
    objective=objective,
    max_iterations=100,
    subset_size=1500
)
```

### Verifying Ray Uniqueness
Rays must be checked for S7 symmetry duplicates using signature comparison:
```python
def get_ray_signature(ray):
    ray_norm = ray / np.linalg.norm(ray)
    unique_vals, counts = np.unique(np.round(ray_norm, 8), return_counts=True)
    return tuple(zip(unique_vals, counts))
```

## Common Issues and Solutions

### LP Infeasibility
- Try different normalizing vectors
- Reduce subset_size to start with fewer constraints
- Check constraint file format and directions

### GPU Memory Issues
- Reduce chunk_size to 100000-300000
- Use `use_gpu=False` for CPU fallback
- Monitor with `nvidia-smi`

### Performance
- Ensure GPU is detected (check startup messages)
- Increase subset_size to reduce iterations but slower LP solves
- Use phase3_no_limit.py for production searches