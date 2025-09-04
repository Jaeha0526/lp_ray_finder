#!/usr/bin/env python3
"""
Use the existing Phase 3 GPU-accelerated active-set implementation 
with the full S₇ constraints
"""

from lp_ray_finder_phase3 import HybridLPRayFinder
import numpy as np
import time

def search_with_phase3():
    """Use Phase 3 implementation to search for new rays"""
    print("🎯 Using Phase 3 GPU-Accelerated Active-Set Method")
    print("=" * 60)
    
    # Load existing rays for comparison
    existing_rays = np.loadtxt('/workspace/lrslib-entropycone/n6data/rays.txt')
    print(f"Loaded {len(existing_rays)} existing rays")
    
    # Initialize with S₇ constraint file
    s7_file = '/workspace/lrslib-entropycone/n6data/n6_correct_s7_expansion.ine'
    
    print(f"\n📁 Loading S₇ constraints from {s7_file}")
    print("Note: Phase 3 will load ALL 8.6M constraints to GPU for violation checking")
    
    # Create finder with GPU acceleration
    finder = HybridLPRayFinder(
        constraint_file=s7_file,
        verbose=True,
        use_gpu=True,
        chunk_size=500000  # Process in chunks for GPU memory
    )
    
    if finder.constraints is None:
        print("❌ Failed to load constraints")
        return
    
    print(f"\n✅ Loaded {len(finder.constraints)} constraints")
    
    # Test 1: Try to recover ray #1381
    print("\n1️⃣ Test: Recover ray #1381 with active-set method")
    target_ray = existing_rays[1380]  # 0-indexed
    objective = target_ray / np.linalg.norm(target_ray)
    
    print(f"Target ray #1381: {target_ray[:10]}... (first 10)")
    
    start = time.time()
    ray, stats = finder.find_extreme_ray(
        objective=objective,
        max_iterations=50,
        subset_size=1000  # Start with 1000 active constraints
    )
    elapsed = time.time() - start
    
    if ray is not None and stats['converged']:
        print(f"\n✅ Found ray in {elapsed:.1f} seconds!")
        print(f"   Iterations: {stats['iterations']}")
        print(f"   Final active set: {len(active_indices)} constraints")
        print(f"   GPU speedup: {stats.get('gpu_speedup', 0):.1f}x")
        
        # Check if it's ray #1381
        ray_scaled = ray * (target_ray[0] / ray[0]) if ray[0] > 0 else ray
        distance = np.linalg.norm(ray_scaled - target_ray)
        print(f"   Distance from target: {distance:.6f}")
    else:
        print(f"\n❌ Failed to find ray")
    
    # Test 2: Search for new rays
    print("\n\n2️⃣ Searching for new rays...")
    
    new_rays = []
    attempts = 5  # Limited due to time
    
    for i in range(attempts):
        print(f"\n{'='*40}")
        print(f"Attempt {i+1}/{attempts}")
        
        # Random objective
        objective = np.random.rand(63)
        objective = objective / np.linalg.norm(objective)
        
        ray, stats = finder.find_extreme_ray(
            objective=objective,
            max_iterations=30,
            subset_size=500
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
                print(f"🎉 FOUND NEW RAY! Total: {len(new_rays)}")
                print(f"   Converged in {stats['iterations']} iterations")
                if stats['gpu_available']:
                    print(f"   GPU speedup: {stats['gpu_speedup']:.1f}x")
            else:
                print(f"Found existing ray")
        else:
            print(f"No convergence or LP failed")
    
    print(f"\n\n📊 RESULTS:")
    print(f"   New rays found: {len(new_rays)}")
    print(f"   GPU acceleration: {'ENABLED' if finder.constraint_manager.use_gpu else 'DISABLED'}")
    
    if len(new_rays) > 0:
        print("\n🎉 SUCCESS! Found new rays with full S₇ validation!")
    else:
        print("\n💡 No new rays found - existing set appears complete")

if __name__ == "__main__":
    # Make sure we're in the right directory
    import os
    os.chdir('/workspace/lrslib-entropycone/lp_ray_finder')
    
    search_with_phase3()