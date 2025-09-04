# 🚀 Quick Start Guide for Future Agents

## What This Project Achieved
We discovered 129 new extreme rays for the N=6 holographic entropy cone, increasing the known count from 4,155 to 4,284 (3.10% increase).

## If You Want to Find More Rays

### 1. Set Up Environment
```bash
cd /workspace/lrslib-entropycone/lp_ray_finder
source venv/bin/activate  # If virtual environment exists
pip install numpy scipy cupy-cuda11x  # GPU support
```

### 2. Run a Search
```python
# Example: Run a small search
python extended_search_5000.py
```

### 3. Analyze Results
```python
# Remove S₇ duplicates
python analyze_extended_search_s7.py

# Check against known rays
python check_against_known_rays.py
```

## Key Files to Understand

### Core Algorithm
- `phase3_no_limit.py` - The LP solver that found the rays
  - Class: `FullConstraintRayFinder`
  - Key method: `find_extreme_ray_active_set()`

### Results
- `truly_new_rays_final.txt` - The 129 new rays
- `all_unique_rays_integer.txt` - Integer representations

### Verification
- `check_against_known_rays.py` - Ensures rays are truly new

## Important Numbers
- Constraints: 8,665,853 (full S₇ expansion)
- Dimensions: 63
- S₇ group size: 5,040 permutations
- Known rays: 4,155 → 4,284 (now)

## Common Tasks

### To Search for More Rays
```python
from phase3_no_limit import FullConstraintRayFinder

finder = FullConstraintRayFinder(
    constraint_file='/workspace/lrslib-entropycone/n6data/n6_correct_s7_expansion.ine'
)

# Try random objectives
new_rays = []
for i in range(1000):
    ray = finder.find_random_ray()
    if ray is not None:
        new_rays.append(ray)
```

### To Check if a Ray is New
```python
import numpy as np

def get_ray_signature(ray):
    ray_norm = ray / np.linalg.norm(ray)
    unique_vals, counts = np.unique(np.round(ray_norm, 8), return_counts=True)
    return tuple(zip(unique_vals, counts))

# Compare signatures with known rays
```

## Tips for Success
1. **Use GPU**: CPU-only is 100x slower
2. **Save frequently**: Searches can run for hours
3. **Expect duplicates**: ~98% of found rays are duplicates
4. **Check memory**: Need ~15GB RAM
5. **Be patient**: Each LP can take 1-30 seconds

## What NOT to Do
❌ Don't try direct vertex enumeration (too slow)
❌ Don't use small constraint subsets (misses rays)
❌ Don't forget to check against known rays
❌ Don't assume rays are new without S₇ analysis

---
*Quick start guide for holographic entropy cone ray discovery*
*Last updated: 2025-08-03*