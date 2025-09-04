#!/usr/bin/env python3
"""
Find new rays in the N=6 holographic entropy cone using LP-based approach
Uses infinity norm normalization instead of sum normalization to avoid infeasibility
"""

import numpy as np
from scipy.optimize import linprog
import time
import os

def load_existing_rays(filename="/workspace/lrslib-entropycone/n6data/rays.txt"):
    """Load the 4,145 existing rays"""
    rays = np.loadtxt(filename)
    print(f"Loaded {len(rays)} existing rays from {filename}")
    return rays

def load_n6_constraints(filename="/workspace/lrslib-entropycone/n6data/n6_correct_s7_expansion.ine", max_constraints=None):
    """Load N=6 entropy cone constraints"""
    print(f"Loading constraints from {filename}...")
    
    constraints = []
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    in_constraints = False
    found_dimensions = False
    m, d = 0, 0
    
    for line in lines:
        line = line.strip()
        
        if not line or line.startswith('#'):
            continue
            
        if 'begin' in line:
            in_constraints = True
            continue
            
        if line == 'end':
            break
            
        if in_constraints and not found_dimensions:
            parts = line.split()
            if len(parts) >= 2 and parts[0].isdigit():
                m = int(parts[0])
                d = int(parts[1])
                found_dimensions = True
                print(f"Matrix dimensions: {m} constraints × {d} variables")
                continue
        
        if in_constraints and found_dimensions:
            try:
                coeffs = [float(x) for x in line.split()]
                if len(coeffs) == d + 1:  # constant + d variables
                    constraints.append(coeffs)
                    
                    if max_constraints and len(constraints) >= max_constraints:
                        break
                        
            except ValueError:
                continue
    
    constraints = np.array(constraints)
    print(f"Loaded {len(constraints)} constraints")
    return constraints, d

def find_ray_with_infinity_norm(A, objective, verbose=True):
    """
    Find extreme ray using infinity norm: max c^T x subject to Ax <= 0, ||x||_∞ = 1
    This is reformulated as: max c^T x subject to Ax <= 0, -1 <= x_i <= 1 for all i
    """
    m, n = A.shape
    
    # LP formulation with bounds
    c = objective
    A_ub = A
    b_ub = np.zeros(m)
    bounds = [(-1, 1) for _ in range(n)]  # -1 <= x_i <= 1
    
    # Solve LP
    result = linprog(-c, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                    method='highs', options={'disp': False})
    
    if result.success:
        ray = result.x
        # Scale so largest component has magnitude 1
        max_abs = np.max(np.abs(ray))
        if max_abs > 0:
            ray = ray / max_abs
        return ray, result
    else:
        return None, result

def find_new_rays(constraints, existing_rays, num_attempts=100):
    """Search for new rays not in the existing set"""
    d = constraints.shape[1] - 1
    A = constraints[:, 1:]  # Remove constant column
    b = -constraints[:, 0]   # Constants on RHS
    
    # For entropy cones, we need Ax >= b, so we use -A and -b for Ax <= b form
    A_ineq = -A
    
    new_rays = []
    
    print(f"\nSearching for new rays with {num_attempts} random objectives...")
    
    for i in range(num_attempts):
        if i % 10 == 0:
            print(f"  Progress: {i}/{num_attempts} attempts, {len(new_rays)} new rays found", end='\r')
        
        # Generate random objective
        objective = np.random.randn(d)
        objective = objective / np.linalg.norm(objective)
        
        # Find extreme ray
        ray, result = find_ray_with_infinity_norm(A_ineq, objective, verbose=False)
        
        if ray is not None:
            # Normalize to positive first non-zero component
            first_nonzero = np.where(np.abs(ray) > 1e-10)[0]
            if len(first_nonzero) > 0 and ray[first_nonzero[0]] < 0:
                ray = -ray
            
            # Check if this ray is new
            is_new = True
            
            # Compare with existing rays
            for existing in existing_rays:
                # Normalize existing ray for comparison
                existing_norm = existing / np.linalg.norm(existing)
                ray_norm = ray / np.linalg.norm(ray)
                
                # Check if parallel (either same or opposite direction)
                similarity = abs(np.dot(ray_norm, existing_norm))
                if similarity > 0.9999:  # Very close to parallel
                    is_new = False
                    break
            
            # Also check against newly found rays
            if is_new:
                for new_ray in new_rays:
                    new_norm = new_ray / np.linalg.norm(new_ray)
                    ray_norm = ray / np.linalg.norm(ray)
                    similarity = abs(np.dot(ray_norm, new_norm))
                    if similarity > 0.9999:
                        is_new = False
                        break
            
            if is_new:
                new_rays.append(ray)
                print(f"\n✅ Found new ray #{len(new_rays)}!")
                print(f"   First 10 components: {ray[:10]}")
                
                # Scale to integer-like values for display
                scale = 1
                for j in range(len(ray)):
                    if abs(ray[j]) > 1e-10:
                        possible_scale = 1 / abs(ray[j])
                        if possible_scale > scale:
                            scale = possible_scale
                
                scaled_ray = ray * scale
                rounded_ray = np.round(scaled_ray)
                if np.allclose(scaled_ray, rounded_ray, rtol=1e-10):
                    print(f"   Integer form: {rounded_ray[:10].astype(int)}")
    
    print(f"\n\nTotal new rays found: {len(new_rays)}")
    return new_rays

def main():
    """Main ray discovery function"""
    print("🎯 N=6 Holographic Entropy Cone - New Ray Discovery")
    print("=" * 60)
    
    # Load existing rays
    existing_rays = load_existing_rays()
    
    # Load constraints (use subset for testing)
    print("\nLoading N=6 constraints...")
    constraints, d = load_n6_constraints(max_constraints=10000)  # Start with 10K constraints
    
    if constraints is None or len(constraints) == 0:
        print("❌ Failed to load constraints")
        return
    
    print(f"\nConstraint system:")
    print(f"  Constraints: {len(constraints)}")
    print(f"  Dimensions: {d}")
    print(f"  Existing rays: {len(existing_rays)}")
    
    # Search for new rays
    start_time = time.time()
    new_rays = find_new_rays(constraints, existing_rays, num_attempts=200)
    elapsed = time.time() - start_time
    
    print(f"\nSearch completed in {elapsed:.1f} seconds")
    
    if new_rays:
        # Save new rays
        output_file = "new_rays_discovered.txt"
        with open(output_file, 'w') as f:
            f.write(f"# New rays discovered in N=6 holographic entropy cone\n")
            f.write(f"# Found {len(new_rays)} rays not in the existing set of {len(existing_rays)}\n")
            f.write(f"# Each row is a 63-dimensional ray\n\n")
            
            for i, ray in enumerate(new_rays):
                # Try to find integer representation
                scale = 1
                for j in range(len(ray)):
                    if abs(ray[j]) > 1e-10:
                        possible_scale = 1 / abs(ray[j])
                        if possible_scale > scale:
                            scale = possible_scale
                
                scaled_ray = ray * scale
                rounded_ray = np.round(scaled_ray)
                
                if np.allclose(scaled_ray, rounded_ray, rtol=1e-10):
                    # Save as integers
                    f.write(" ".join(f"{int(x)}" for x in rounded_ray) + "\n")
                else:
                    # Save as floats
                    f.write(" ".join(f"{x:.10f}" for x in ray) + "\n")
        
        print(f"\n✅ Saved {len(new_rays)} new rays to {output_file}")
        
        # Verify the new rays satisfy constraints
        print("\nVerifying new rays against constraints...")
        violations = 0
        for ray in new_rays:
            # Check Ax >= b (original form)
            A = constraints[:, 1:]
            b = -constraints[:, 0]
            
            residuals = A @ ray - b
            if np.any(residuals < -1e-10):
                violations += 1
        
        print(f"Constraint violations: {violations}/{len(new_rays)} rays")
        
    else:
        print("\n❌ No new rays found beyond the existing set")
        print("This suggests the existing set of 4,145 rays may be complete")

if __name__ == "__main__":
    main()