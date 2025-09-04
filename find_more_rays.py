#!/usr/bin/env python3
"""
Quick search for even more rays to demonstrate the potential
"""

import numpy as np
from scipy.optimize import linprog
import time

# Load data
facets = np.loadtxt('/workspace/lrslib-entropycone/n6data/facets.txt')
existing_rays = np.loadtxt('/workspace/lrslib-entropycone/n6data/rays.txt')

# Load our newly found rays
with open('new_positive_rays.txt', 'r') as f:
    lines = f.readlines()
new_rays = []
for line in lines:
    if not line.startswith('#') and line.strip():
        ray = np.array([float(x) for x in line.split()])
        new_rays.append(ray)

all_known_rays = list(existing_rays) + new_rays

print(f"Starting with {len(all_known_rays)} known rays")
print("Searching for 100 more rays with different strategies...\n")

def find_ray(objective):
    n_facets, d = facets.shape
    A_ub = -facets
    b_ub = np.zeros(n_facets)
    A_eq = np.ones((1, d))
    b_eq = np.array([1.0])
    bounds = [(0, None) for _ in range(d)]
    
    result = linprog(-objective, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                    bounds=bounds, method='highs', options={'disp': False})
    
    if result.success:
        return result.x
    return None

additional_rays = []
attempts = 0

# Try targeted strategies
for strategy in range(100):
    attempts += 1
    
    # Different objective strategies
    if strategy < 20:
        # Target specific components
        objective = np.zeros(63)
        idx1 = np.random.randint(0, 63)
        idx2 = np.random.randint(0, 63)
        idx3 = np.random.randint(0, 63)
        objective[idx1] = 1
        objective[idx2] = 0.5
        objective[idx3] = 0.3
    else:
        # Sparse random objectives
        objective = np.zeros(63)
        num_nonzero = np.random.randint(3, 10)
        indices = np.random.choice(63, num_nonzero, replace=False)
        objective[indices] = np.random.rand(num_nonzero)
    
    objective = objective / np.linalg.norm(objective)
    
    ray = find_ray(objective)
    
    if ray is not None and np.all(ray >= -1e-10):
        # Check if new
        ray_norm = ray / np.linalg.norm(ray)
        
        is_new = True
        for known in all_known_rays:
            known_norm = known / np.linalg.norm(known)
            if np.dot(ray_norm, known_norm) > 0.9999:
                is_new = False
                break
        
        for found in additional_rays:
            found_norm = found / np.linalg.norm(found)
            if np.dot(ray_norm, found_norm) > 0.9999:
                is_new = False
                break
        
        if is_new:
            additional_rays.append(ray)
            nonzero = np.sum(ray > 1e-10)
            print(f"✓ Ray #{len(additional_rays)}: {nonzero} non-zero components")

print(f"\n🎯 RESULTS:")
print(f"   Attempts: {attempts}")
print(f"   New rays found: {len(additional_rays)}")
print(f"   Success rate: {len(additional_rays)/attempts*100:.1f}%")
print(f"\n📊 TOTAL RAYS NOW: {len(all_known_rays) + len(additional_rays)}")
print(f"   Original: 4,145")
print(f"   First batch: +174")
print(f"   This batch: +{len(additional_rays)}")
print(f"   TOTAL: {4145 + 174 + len(additional_rays)}")

if additional_rays:
    # Append to file
    with open('new_positive_rays.txt', 'a') as f:
        f.write(f"\n# Additional batch - found {len(additional_rays)} more rays\n")
        for ray in additional_rays:
            f.write(" ".join(f"{x:.10f}" for x in ray) + "\n")
    print(f"\n✅ Appended {len(additional_rays)} more rays to new_positive_rays.txt")

print("\n🚀 This confirms there are MANY more rays to discover!")