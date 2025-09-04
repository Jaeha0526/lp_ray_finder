# How to Find New Rays with LP-Based Ray Finder

## Overview

The LP-based ray finder provides a computationally feasible alternative to vertex enumeration for finding extreme rays in the N=6 holographic entropy cone. Instead of enumerating all rays (which would take decades), it finds specific rays through targeted LP optimization.

## Prerequisites

1. **N=6 Constraint File**: You need the actual constraint file with 8.6M constraints
   - Expected location: `../n6data/n6_restart_startingcobasis.ine` or similar
   - Format: Standard .ine format with inequality constraints

2. **GPU Setup**: For optimal performance
   - NVIDIA GPU with CUDA support
   - CuPy installed: `pip install cupy-cuda12x`

## Step-by-Step Guide

### 1. Basic Usage - Find Any Extreme Ray

```python
from lp_ray_finder_phase3 import HybridLPRayFinder
import numpy as np

# Load the N=6 constraint file
finder = HybridLPRayFinder(
    constraint_file="path/to/n6_constraints.ine",
    verbose=True,
    use_gpu=True,
    chunk_size=500000  # Adjust based on GPU memory
)

# Find a ray with random objective
ray, stats = finder.find_extreme_ray()

if ray is not None:
    print(f"Found extreme ray: {ray}")
    print(f"Converged in {stats['iterations']} iterations")
```

### 2. Find Rays Similar to Ray #1381

```python
# Target ray #1381 coordinates
target_ray = [1, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 6, 6, 8, 8, 8, 8, 7, 7, 9, 9, 7, 7, 9, 7, 9, 9, 6, 8, 8, 8, 7, 7, 7, 7, 6, 6, 6, 6, 6, 5, 4, 3]

# Use target as objective direction
objective = np.array(target_ray) / np.linalg.norm(target_ray)

# Find ray in this direction
ray, stats = finder.find_extreme_ray(
    objective=objective,
    max_iterations=100,
    subset_size=1500
)
```

### 3. Discover Multiple New Rays

```python
def find_multiple_rays(finder, num_rays=10):
    """Find multiple diverse extreme rays"""
    rays_found = []
    
    for i in range(num_rays):
        # Random objective for diversity
        objective = np.random.randn(finder.d)
        objective = objective / np.linalg.norm(objective)
        
        ray, stats = finder.find_extreme_ray(objective=objective)
        
        if ray is not None:
            # Check if it's truly new
            is_new = True
            for existing in rays_found:
                if np.abs(np.dot(ray, existing)) > 0.99:  # Nearly parallel
                    is_new = False
                    break
            
            if is_new:
                rays_found.append(ray)
                print(f"Ray {len(rays_found)}: Found in {stats['iterations']} iterations")
    
    return rays_found

# Find 10 new rays
new_rays = find_multiple_rays(finder, num_rays=10)
print(f"Discovered {len(new_rays)} unique extreme rays")
```

### 4. Search in Specific Regions

```python
def find_rays_near_target(finder, target_direction, num_attempts=20, perturbation=0.2):
    """Find rays near a specific direction"""
    rays_found = []
    target_norm = target_direction / np.linalg.norm(target_direction)
    
    for i in range(num_attempts):
        # Perturb the target direction slightly
        noise = np.random.randn(len(target_direction)) * perturbation
        objective = target_norm + noise
        objective = objective / np.linalg.norm(objective)
        
        ray, stats = finder.find_extreme_ray(objective=objective)
        
        if ray is not None and stats['converged']:
            angle = np.arccos(np.clip(np.dot(ray, target_norm), -1, 1))
            angle_deg = np.degrees(angle)
            
            if angle_deg < 30:  # Within 30 degrees
                rays_found.append((ray, angle_deg))
                print(f"Found ray at {angle_deg:.1f}° from target")
    
    return rays_found
```

## Key Parameters

### For `HybridLPRayFinder`:
- **`chunk_size`**: 500000-1000000 for GPU memory management
- **`use_gpu`**: True if GPU available (100-1000x speedup)
- **`verbose`**: True to see progress

### For `find_extreme_ray()`:
- **`objective`**: Direction to optimize (finds ray maximizing c^T x)
- **`max_iterations`**: 50-200 (more for difficult cases)
- **`subset_size`**: 1000-2000 (initial active constraints)

## Expected Performance

With GPU acceleration on the N=6 system:
- **Per ray discovery**: 30 seconds to 5 minutes
- **Memory usage**: ~4-8 GB GPU memory
- **Speedup vs enumeration**: 10,000x or more

## Troubleshooting

### LP Infeasibility
If you get "The problem is infeasible" errors:
1. Check constraint formulation (signs, directions)
2. Try different normalizing vectors
3. Reduce `subset_size` to start with fewer constraints
4. Verify the constraint file format

### GPU Memory Issues
If GPU runs out of memory:
1. Reduce `chunk_size` to 100000-300000
2. Use CPU fallback: `use_gpu=False`
3. Check available memory with `nvidia-smi`

### Slow Performance
1. Ensure GPU is being used (check output messages)
2. Increase `subset_size` to reduce iterations
3. Use Phase 2 (CPU parallel) if GPU unavailable

## Complete Example Script

```python
#!/usr/bin/env python3
"""Find new extreme rays in N=6 holographic entropy cone"""

import numpy as np
from lp_ray_finder_phase3 import HybridLPRayFinder
import time

def main():
    # Load N=6 system
    print("Loading N=6 holographic entropy cone...")
    finder = HybridLPRayFinder(
        constraint_file="../n6data/n6_restart_startingcobasis.ine",
        verbose=True,
        use_gpu=True,
        chunk_size=500000
    )
    
    if finder.constraints is None:
        print("Failed to load constraints!")
        return
    
    print(f"Loaded {len(finder.constraints):,} constraints in {finder.d} dimensions")
    
    # Find 5 new rays
    rays = []
    for i in range(5):
        print(f"\nSearching for ray {i+1}...")
        start = time.time()
        
        # Random objective
        objective = np.random.randn(finder.d)
        objective = objective / np.linalg.norm(objective)
        
        ray, stats = finder.find_extreme_ray(
            objective=objective,
            max_iterations=100,
            subset_size=1500
        )
        
        if ray is not None:
            rays.append(ray)
            elapsed = time.time() - start
            print(f"✓ Found ray in {elapsed:.1f}s ({stats['iterations']} iterations)")
            print(f"  First 10 components: {ray[:10]}")
        else:
            print("✗ No ray found")
    
    print(f"\nTotal rays discovered: {len(rays)}")
    
    # Save rays for analysis
    if rays:
        np.save("discovered_rays.npy", np.array(rays))
        print("Saved rays to discovered_rays.npy")

if __name__ == "__main__":
    main()
```

## Summary

The LP-based ray finder enables:
- **Targeted ray discovery** without full enumeration
- **Minutes to hours** instead of decades for results
- **GPU acceleration** for massive speedup
- **Scalability** to millions of constraints

This approach makes the N=6 holographic entropy cone computationally accessible for the first time!