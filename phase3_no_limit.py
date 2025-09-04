#!/usr/bin/env python3
"""
Phase 3 WITHOUT the 100K constraint limit - loads ALL S₇ constraints
"""

import numpy as np
from scipy.optimize import linprog
import time
import random
import gc

class FullConstraintRayFinder:
    """Ray finder that loads ALL constraints"""
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.constraints = None
        self.d = 63
        
    def load_all_s7_constraints(self, filename):
        """Load ALL S₇ constraints without limits"""
        if self.verbose:
            print(f"📁 Loading ALL constraints from {filename}")
            print("⚠️  This will load 8.6M constraints (~4GB RAM)")
        
        start_time = time.time()
        constraints = []
        
        with open(filename, 'r') as f:
            in_data = False
            count = 0
            
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
                        # S₇ format: 63 coefficients, no constant
                        facet = [float(x) for x in values]
                        constraints.append(facet)
                        count += 1
                        
                        if count % 1000000 == 0:
                            print(f"   Loaded {count:,} constraints...")
        
        self.constraints = np.array(constraints)
        elapsed = time.time() - start_time
        
        print(f"✅ Loaded {len(self.constraints):,} constraints in {elapsed:.1f}s")
        print(f"📊 Memory: {self.constraints.nbytes / 1024**3:.2f} GB")
        
        return len(self.constraints)
    
    def find_ray_active_set(self, objective, initial_size=1000, max_iter=100):
        """Active-set method with ALL constraints available for checking"""
        if self.verbose:
            print(f"\n🎯 Active-set search (max {max_iter} iterations)")
        
        n = self.d
        total_constraints = len(self.constraints)
        
        # Start with random subset
        active_indices = random.sample(range(total_constraints), 
                                     min(initial_size, total_constraints))
        
        for iteration in range(max_iter):
            # Get active constraints
            active_constraints = self.constraints[active_indices]
            
            # Solve LP with active set only
            # Try without normalization constraint first
            m = len(active_constraints)
            
            # Just the cone constraints: Ax >= 0
            A_ub = -active_constraints
            b_ub = np.zeros(m)
            
            # Add bounds to prevent unbounded solution
            bounds = [(0, 1) for _ in range(n)]
            
            try:
                result = linprog(
                    -objective,  # Maximize
                    A_ub=A_ub,
                    b_ub=b_ub,
                    bounds=bounds,
                    method='highs',
                    options={'disp': False}
                )
                
                if not result.success:
                    if self.verbose and iteration == 0:
                        print(f"   Iteration {iteration+1}: LP failed - {result.message}")
                    return None, False
                
                ray = result.x
                
                # Normalize ray
                ray_norm = np.linalg.norm(ray)
                if ray_norm > 0:
                    ray = ray / ray_norm
                
                # Check violations against ALL constraints
                violations = self.constraints @ ray
                violated_indices = np.where(violations < -1e-10)[0]
                
                if self.verbose and iteration % 10 == 0:
                    print(f"   Iteration {iteration+1}: {len(active_indices)} active, "
                          f"{len(violated_indices)} violations")
                
                if len(violated_indices) == 0:
                    if self.verbose:
                        print(f"✅ Converged in {iteration+1} iterations!")
                    return ray, True
                
                # Add most violated
                violation_amounts = -violations[violated_indices]
                most_violated = violated_indices[np.argsort(violation_amounts)[-50:]]
                
                new_constraints = [idx for idx in most_violated if idx not in active_indices]
                active_indices.extend(new_constraints[:50])  # Add up to 50
                
            except Exception as e:
                if self.verbose:
                    print(f"   Error in iteration {iteration+1}: {e}")
                return None, False
        
        if self.verbose:
            print(f"⚠️  Didn't converge in {max_iter} iterations")
        return ray, False

def main():
    print("🚀 Phase 3 with FULL S₇ Constraints (No 100K Limit)")
    print("=" * 60)
    
    # Load existing rays
    existing_rays = np.loadtxt('/workspace/lrslib-entropycone/n6data/rays.txt')
    print(f"Loaded {len(existing_rays)} existing rays")
    
    # Create finder
    finder = FullConstraintRayFinder(verbose=True)
    
    # Load ALL S₇ constraints
    s7_file = '/workspace/lrslib-entropycone/n6data/n6_correct_s7_expansion.ine'
    num_constraints = finder.load_all_s7_constraints(s7_file)
    
    if num_constraints < 8000000:
        print(f"⚠️  Warning: Only loaded {num_constraints:,} constraints")
        print("   Expected ~8.6M constraints")
    
    # Save new rays for debugging
    new_rays_file = open('new_rays_full_s7_debug.txt', 'w')
    
    # Search for rays
    print("\n🔍 Searching for new rays with FULL constraint set...")
    
    attempts = 10
    new_rays = []
    
    for i in range(attempts):
        print(f"\n{'='*40}")
        print(f"Attempt {i+1}/{attempts}")
        
        # Random objective
        objective = np.random.rand(63)
        objective = objective / np.linalg.norm(objective)
        
        ray, converged = finder.find_ray_active_set(objective, initial_size=500, max_iter=50)
        
        if ray is not None and converged:
            # Check if new
            is_new = True
            for existing in existing_rays:
                existing_norm = existing / np.linalg.norm(existing) 
                if np.dot(ray, existing_norm) > 0.9999:
                    is_new = False
                    break
            
            if is_new:
                new_rays.append(ray)
                print(f"🎉 NEW RAY FOUND! Total: {len(new_rays)}")
                # Save for debugging
                np.savetxt(new_rays_file, [ray], fmt='%.10f')
                new_rays_file.flush()
                
                # Also check constraint violations
                print(f"   Checking full S₇ violations...")
                violations = finder.constraints @ ray
                num_violations = np.sum(violations < -1e-10)
                print(f"   Violations: {num_violations} out of {len(finder.constraints)}")
            else:
                print(f"Found existing ray")
        else:
            print(f"No convergence or LP failed")
    
    new_rays_file.close()
    
    print(f"\n📊 RESULTS with {num_constraints:,} S₇ constraints:")
    print(f"   New rays found: {len(new_rays)}")
    
    if len(new_rays) > 0:
        print("\n🎉 BREAKTHROUGH: Found new rays with FULL S₇ constraints!")
        print("   Saved to: new_rays_full_s7_debug.txt")
    else:
        print("\n✅ No new rays found - confirms existing set is complete")

if __name__ == "__main__":
    main()