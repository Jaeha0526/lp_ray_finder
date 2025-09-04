#!/usr/bin/env python3
"""
Find rays using active-set method with full S₇ constraints
This is the efficient approach that avoids loading all 8.6M constraints into the LP solver
"""

import numpy as np
from scipy.optimize import linprog
import time
import random

def load_s7_constraints_streaming(filename):
    """
    Generator to stream S₇ constraints without loading all into memory at once
    """
    with open(filename, 'r') as f:
        in_data = False
        for line in f:
            line = line.strip()
            
            if line == 'begin':
                in_data = True
                continue
            elif line == 'end':
                break
            elif in_data and line and (line[0].isdigit() or line[0] == '-'):
                if 'integer' in line:
                    continue
                
                values = line.split()
                if len(values) == 63:
                    yield np.array([float(x) for x in values])

def check_violations_batch(ray, constraint_file, batch_size=100000):
    """
    Check which constraints are violated by the ray
    Returns indices of violated constraints
    """
    violations = []
    constraint_idx = 0
    
    for constraint in load_s7_constraints_streaming(constraint_file):
        if np.dot(constraint, ray) < -1e-10:
            violations.append(constraint_idx)
        
        constraint_idx += 1
        
        # Return early if we have enough violations
        if len(violations) >= batch_size:
            break
    
    return violations, constraint_idx

def active_set_ray_finder(constraint_file, objective, initial_subset_size=1000, max_iterations=50, verbose=True):
    """
    Find extreme ray using active-set method
    
    1. Start with random subset of constraints
    2. Solve LP
    3. Check violations against ALL constraints
    4. Add violated constraints to active set
    5. Repeat until convergence
    """
    
    if verbose:
        print(f"\n🎯 Active-set method with S₇ constraints")
        print(f"   Initial subset: {initial_subset_size}")
        print(f"   Max iterations: {max_iterations}")
    
    # Load initial random subset
    if verbose:
        print(f"\n1️⃣ Loading initial subset of constraints...")
    
    all_constraints = []
    for i, constraint in enumerate(load_s7_constraints_streaming(constraint_file)):
        all_constraints.append(constraint)
        if i >= initial_subset_size * 10:  # Load more than needed for random selection
            break
    
    # Random initial active set
    active_indices = random.sample(range(len(all_constraints)), min(initial_subset_size, len(all_constraints)))
    active_constraints = [all_constraints[i] for i in active_indices]
    
    converged = False
    iteration = 0
    
    while iteration < max_iterations and not converged:
        iteration += 1
        
        if verbose:
            print(f"\n🔄 Iteration {iteration}: {len(active_constraints)} active constraints")
        
        # Solve LP with active constraints
        n = 63
        m = len(active_constraints)
        
        # Set up LP: max c^T x s.t. Ax >= 0, sum(x) = 1, x >= 0
        A_ub = -np.array(active_constraints)
        b_ub = np.zeros(m)
        A_eq = np.ones((1, n))
        b_eq = np.array([1.0])
        bounds = [(0, None) for _ in range(n)]
        
        result = linprog(
            -objective,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method='highs',
            options={'disp': False}
        )
        
        if not result.success:
            if verbose:
                print(f"   ❌ LP failed: {result.message}")
            return None, {'converged': False, 'iterations': iteration}
        
        ray = result.x
        
        # Check violations against ALL S₇ constraints
        if verbose:
            print(f"   🔍 Checking violations against all S₇ constraints...")
        
        violations, total_checked = check_violations_batch(ray, constraint_file, batch_size=1000)
        
        if verbose:
            print(f"   📊 Found {len(violations)} violations (checked {total_checked} constraints)")
        
        if len(violations) == 0:
            # No violations - we found a valid ray!
            converged = True
            if verbose:
                print(f"   ✅ Converged! No violations found.")
        else:
            # Add violated constraints
            # Load the specific violated constraints
            new_constraints = []
            for i, constraint in enumerate(load_s7_constraints_streaming(constraint_file)):
                if i in violations:
                    new_constraints.append(constraint)
                if len(new_constraints) >= len(violations):
                    break
            
            active_constraints.extend(new_constraints)
            
            if verbose:
                print(f"   ➕ Added {len(new_constraints)} constraints to active set")
    
    stats = {
        'converged': converged,
        'iterations': iteration,
        'final_active_size': len(active_constraints)
    }
    
    if converged:
        return ray, stats
    else:
        if verbose:
            print(f"   ⚠️  Did not converge in {max_iterations} iterations")
        return ray, stats  # Return best ray found

def search_with_active_set():
    """
    Search for new rays using active-set method
    """
    print("🎯 Searching for rays with active-set method on S₇ constraints")
    print("=" * 70)
    
    # Load existing rays
    existing_rays = np.loadtxt('/workspace/lrslib-entropycone/n6data/rays.txt')
    print(f"Loaded {len(existing_rays)} existing rays")
    
    constraint_file = '/workspace/lrslib-entropycone/n6data/n6_correct_s7_expansion.ine'
    
    # Test 1: Try to recover ray #1381
    print("\n1️⃣ Testing: Recover ray #1381")
    target_ray = existing_rays[1380]  # 0-indexed
    objective = target_ray / np.linalg.norm(target_ray)
    
    ray, stats = active_set_ray_finder(
        constraint_file, 
        objective,
        initial_subset_size=500,
        max_iterations=20,
        verbose=True
    )
    
    if ray is not None and stats['converged']:
        ray_scaled = ray * (target_ray[0] / ray[0]) if ray[0] > 0 else ray
        distance = np.linalg.norm(ray_scaled - target_ray)
        print(f"\n📏 Distance from target ray #1381: {distance:.6f}")
        
        if distance < 0.01:
            print("✅ Successfully recovered ray #1381 with active-set method!")
    
    # Test 2: Search for new rays
    print("\n\n2️⃣ Searching for new rays...")
    
    new_rays = []
    attempts = 10  # Limited due to computational cost
    
    for i in range(attempts):
        print(f"\n{'='*50}")
        print(f"Attempt {i+1}/{attempts}")
        
        # Random objective
        objective = np.random.rand(63)
        objective = objective / np.linalg.norm(objective)
        
        ray, stats = active_set_ray_finder(
            constraint_file,
            objective,
            initial_subset_size=500,
            max_iterations=20,
            verbose=False
        )
        
        if ray is not None and stats['converged']:
            # Check if new
            is_new = True
            ray_norm = ray / np.linalg.norm(ray)
            
            for existing in existing_rays:
                existing_norm = existing / np.linalg.norm(existing)
                if np.dot(ray_norm, existing_norm) > 0.9999:
                    is_new = False
                    break
            
            if is_new:
                new_rays.append(ray)
                print(f"  🎉 Found NEW ray! Total: {len(new_rays)}")
                print(f"     Converged in {stats['iterations']} iterations")
                print(f"     Active set size: {stats['final_active_size']}")
            else:
                print(f"  - Found existing ray (converged in {stats['iterations']} iterations)")
        else:
            print(f"  - No convergence")
    
    print(f"\n\n📊 FINAL RESULTS:")
    print(f"   Attempts: {attempts}")
    print(f"   New rays found: {len(new_rays)}")
    
    if len(new_rays) > 0:
        print("\n🎉 Found new rays with full S₇ validation!")
    else:
        print("\n💡 No new rays found - existing set appears complete")

if __name__ == "__main__":
    search_with_active_set()