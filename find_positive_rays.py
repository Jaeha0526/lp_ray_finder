#!/usr/bin/env python3
"""
Find new rays in the positive orthant of the N=6 holographic entropy cone
"""

import numpy as np
from scipy.optimize import linprog
import time

def load_data():
    """Load existing facets and rays"""
    print("Loading existing data...")
    facets = np.loadtxt('/workspace/lrslib-entropycone/n6data/facets.txt')
    rays = np.loadtxt('/workspace/lrslib-entropycone/n6data/rays.txt')
    print(f"Loaded {len(facets)} facets and {len(rays)} rays")
    return facets, rays

def find_positive_ray_lp(facets, objective):
    """
    Find extreme ray in positive orthant
    Maximize: objective · ray
    Subject to: facet_i · ray >= 0 for all i, ray_j >= 0 for all j, sum(ray) = 1
    """
    n_facets, d = facets.shape
    
    # Constraints: -facet · ray <= 0 (i.e., facet · ray >= 0)
    A_ub = -facets
    b_ub = np.zeros(n_facets)
    
    # Normalization: sum(ray) = 1
    A_eq = np.ones((1, d))
    b_eq = np.array([1.0])
    
    # Bounds: ray_j >= 0 (non-negative)
    bounds = [(0, None) for _ in range(d)]
    
    # Solve LP
    result = linprog(
        -objective,  # Maximize
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method='highs',
        options={'disp': False}
    )
    
    if result.success:
        return result.x
    else:
        return None

def scale_to_integers(ray, max_denominator=1000):
    """Try to find integer representation of ray"""
    # Find smallest positive component
    positive = ray[ray > 1e-10]
    if len(positive) == 0:
        return ray
    
    min_positive = np.min(positive)
    
    # Try different scales
    for denom in range(1, max_denominator):
        scaled = ray / min_positive * denom
        rounded = np.round(scaled)
        if np.allclose(scaled, rounded, rtol=1e-10, atol=1e-10):
            return rounded.astype(int)
    
    return ray

def search_for_new_positive_rays():
    """Search for new rays in positive orthant"""
    facets, rays = load_data()
    
    print("\n🔍 Searching for new rays in positive orthant...")
    
    new_rays = []
    num_attempts = 200
    
    # Different objective strategies
    for i in range(num_attempts):
        if i % 20 == 0:
            print(f"   Progress: {i}/{num_attempts} attempts, {len(new_rays)} new rays found", end='\r')
        
        # Generate objective
        if i < 63:
            # Try unit vectors
            objective = np.zeros(63)
            objective[i] = 1
        elif i < 126:
            # Try pairs
            objective = np.zeros(63)
            idx1, idx2 = i - 63, (i - 63 + 1) % 63
            objective[idx1] = 1
            objective[idx2] = 1
        else:
            # Random positive objective
            objective = np.random.rand(63)
        
        objective = objective / np.linalg.norm(objective)
        
        found_ray = find_positive_ray_lp(facets, objective)
        
        if found_ray is not None:
            # Check if positive
            if np.all(found_ray >= -1e-10):
                # Check if new
                is_new = True
                
                # Normalize for comparison
                found_norm = found_ray / np.linalg.norm(found_ray)
                
                for existing_ray in rays:
                    existing_norm = existing_ray / np.linalg.norm(existing_ray)
                    similarity = np.dot(found_norm, existing_norm)
                    if similarity > 0.9999:
                        is_new = False
                        break
                
                for new_ray in new_rays:
                    new_norm = new_ray / np.linalg.norm(new_ray)
                    similarity = np.dot(found_norm, new_norm)
                    if similarity > 0.9999:
                        is_new = False
                        break
                
                if is_new:
                    new_rays.append(found_ray)
                    print(f"\n   ✅ Found new positive ray #{len(new_rays)}!")
                    
                    # Try to find integer form
                    int_ray = scale_to_integers(found_ray)
                    if isinstance(int_ray, np.ndarray) and int_ray.dtype == np.int64:
                        print(f"      Integer form: {int_ray[:10]}... (first 10)")
                    else:
                        print(f"      First 10: {found_ray[:10]}")
                    
                    # Check sparsity
                    nonzero = np.sum(found_ray > 1e-10)
                    print(f"      Non-zero components: {nonzero}/63")
    
    print(f"\n\n📊 Final Summary:")
    print(f"   Existing rays: {len(rays)}")
    print(f"   New positive rays found: {len(new_rays)}")
    
    if len(new_rays) > 0:
        # Save results
        output_file = "new_positive_rays.txt"
        with open(output_file, 'w') as f:
            f.write(f"# New positive rays discovered in N=6 holographic entropy cone\n")
            f.write(f"# Found {len(new_rays)} rays not in the existing set of {len(rays)}\n")
            f.write(f"# Each row is a 63-dimensional ray\n\n")
            
            for i, ray in enumerate(new_rays):
                int_ray = scale_to_integers(ray)
                if isinstance(int_ray, np.ndarray) and int_ray.dtype == np.int64:
                    f.write(" ".join(str(x) for x in int_ray) + "\n")
                else:
                    f.write(" ".join(f"{x:.10f}" for x in ray) + "\n")
        
        print(f"   ✅ Saved new rays to {output_file}")
        
        # Analyze structure
        print("\n   Ray structure analysis:")
        for i in range(min(5, len(new_rays))):
            ray = new_rays[i]
            nonzero = np.sum(ray > 1e-10)
            max_comp = np.max(ray)
            print(f"   Ray {i+1}: {nonzero} non-zero, max = {max_comp:.3f}")
    
    return new_rays

if __name__ == "__main__":
    new_rays = search_for_new_positive_rays()