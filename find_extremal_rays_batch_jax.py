#!/usr/bin/env python3
"""
Batch script for finding extremal rays using JAX-accelerated implementation
Optimized for maximum GPU performance on cluster
"""

import numpy as np
import argparse
import os
import sys
import time
import json
from datetime import datetime
from pathlib import Path

def setup_directories(output_dir, job_id):
    """Create output directories for results"""
    base_dir = Path(output_dir)
    job_dir = base_dir / f"job_{job_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    job_dir.mkdir(parents=True, exist_ok=True)
    
    rays_dir = job_dir / "rays"
    rays_dir.mkdir(exist_ok=True)
    
    return job_dir, rays_dir

def save_ray(ray, stats, index, rays_dir):
    """Save individual ray with metadata"""
    ray_file = rays_dir / f"ray_{index:04d}.npz"
    np.savez(ray_file, 
             ray=ray,
             iterations=stats.get('iterations', -1),
             time=stats.get('total_time', -1),
             converged=stats.get('converged', False),
             rank=stats.get('rank', -1),
             gpu_time=stats.get('gpu_time', -1),
             lp_time=stats.get('lp_time', -1))
    return ray_file

def get_ray_signature(ray):
    """Get signature for duplicate detection"""
    ray_norm = ray / np.linalg.norm(ray)
    unique_vals, counts = np.unique(np.round(ray_norm, 8), return_counts=True)
    return tuple(zip(unique_vals, counts))

def find_extremal_rays_jax(constraint_file, num_searches, output_dir, job_id, 
                           checkpoint_interval=10, use_jax=True):
    """Main function to find extremal rays using JAX or fallback to Phase3"""
    
    # Setup directories
    job_dir, rays_dir = setup_directories(output_dir, job_id)
    
    # Initialize log file
    log_file = job_dir / "search_log.txt"
    summary_file = job_dir / "summary.json"
    
    print(f"Job directory: {job_dir}")
    print(f"Loading constraints from: {constraint_file}")
    
    # Try to use JAX version, fallback to Phase3 if needed
    finder = None
    using_jax = False
    
    if use_jax:
        try:
            from lp_ray_finder_jax import JAXRayFinder
            print("🚀 Using JAX-accelerated implementation")
            finder = JAXRayFinder(
                constraint_file=constraint_file,
                use_float32=True,  # Faster with minimal precision loss
                verbose=True
            )
            using_jax = True
            
            if finder.constraints_gpu is None:
                print("ERROR: Failed to load constraints with JAX!")
                finder = None
                
        except Exception as e:
            print(f"⚠️ JAX initialization failed: {e}")
            print("Falling back to CuPy/Phase3 implementation...")
    
    # Fallback to Phase3 implementation
    if finder is None:
        try:
            from lp_ray_finder_phase3 import HybridLPRayFinder
            print("🎮 Using Phase3 CuPy/CPU implementation (fallback)")
            finder = HybridLPRayFinder(
                constraint_file=constraint_file,
                verbose=True,
                use_gpu=True,
                chunk_size=500000
            )
            
            if finder.constraints is None:
                print("ERROR: Failed to load constraints!")
                return 1
                
        except Exception as e:
            print(f"ERROR: Could not initialize any ray finder: {e}")
            return 1
    
    # Get dimensions
    d = finder.d
    n_constraints = len(finder.constraints_gpu if using_jax else finder.constraints)
    
    print(f"Loaded {n_constraints:,} constraints in {d} dimensions")
    print(f"Implementation: {'JAX GPU' if using_jax else 'CuPy/CPU'}")
    
    # Track results
    results = {
        'job_id': job_id,
        'start_time': datetime.now().isoformat(),
        'constraint_file': constraint_file,
        'num_constraints': n_constraints,
        'dimensions': d,
        'num_searches': num_searches,
        'implementation': 'JAX' if using_jax else 'Phase3',
        'rays_found': [],
        'unique_signatures': set(),
        'statistics': {
            'total_found': 0,
            'unique_found': 0,
            'converged': 0,
            'failed': 0,
            'total_time': 0,
            'total_gpu_time': 0,
            'total_lp_time': 0
        }
    }
    
    # Performance tracking
    search_times = []
    
    # Main search loop
    with open(log_file, 'w') as log:
        log.write(f"Starting search at {datetime.now()}\n")
        log.write(f"Implementation: {'JAX GPU' if using_jax else 'Phase3'}\n")
        log.write(f"Constraints: {n_constraints:,}, Dimensions: {d}\n\n")
        
        for i in range(num_searches):
            print(f"\n[{i+1}/{num_searches}] Searching for extremal ray...")
            log.write(f"\n[{i+1}/{num_searches}] Search started at {datetime.now()}\n")
            
            start_time = time.time()
            
            try:
                # Generate random objective
                if using_jax:
                    import jax.numpy as jnp
                    import jax
                    key = jax.random.PRNGKey(int(time.time() * 1000) + i)
                    objective = jax.random.normal(key, (d,), dtype=jnp.float32)
                    objective = objective / jnp.linalg.norm(objective)
                else:
                    objective = np.random.randn(d)
                    objective = objective / np.linalg.norm(objective)
                
                # Find extremal ray with verification
                ray, stats = finder.find_extreme_ray(
                    objective=objective,
                    max_iterations=100,
                    subset_size=1500
                )
                
                elapsed = time.time() - start_time
                search_times.append(elapsed)
                results['statistics']['total_time'] += elapsed
                
                if stats.get('gpu_time', 0) > 0:
                    results['statistics']['total_gpu_time'] += stats['gpu_time']
                if stats.get('lp_time', 0) > 0:
                    results['statistics']['total_lp_time'] += stats['lp_time']
                
                if ray is not None and stats.get('converged', False):
                    # Check extremality (rank should be d-1)
                    expected_rank = d - 1
                    actual_rank = stats.get('rank', 0)
                    
                    if actual_rank >= expected_rank - 1:  # Allow small tolerance
                        # Save ray
                        ray_file = save_ray(ray, stats, i, rays_dir)
                        
                        # Check for uniqueness
                        signature = get_ray_signature(ray)
                        is_new = signature not in results['unique_signatures']
                        
                        if is_new:
                            results['unique_signatures'].add(signature)
                            results['statistics']['unique_found'] += 1
                        
                        results['statistics']['total_found'] += 1
                        results['statistics']['converged'] += 1
                        
                        # Log success
                        log.write(f"  ✓ Found extremal ray (rank {actual_rank}/{expected_rank})\n")
                        log.write(f"    Iterations: {stats.get('iterations', -1)}\n")
                        log.write(f"    Time: {elapsed:.2f}s\n")
                        log.write(f"    GPU time: {stats.get('gpu_time', 0):.2f}s\n")
                        log.write(f"    LP time: {stats.get('lp_time', 0):.2f}s\n")
                        log.write(f"    Unique: {'Yes' if is_new else 'No'}\n")
                        log.write(f"    File: {ray_file.name}\n")
                        
                        print(f"  ✓ Found extremal ray in {elapsed:.1f}s "
                              f"(iter: {stats.get('iterations', -1)}, "
                              f"unique: {'Yes' if is_new else 'No'})")
                        
                        # Store ray info
                        results['rays_found'].append({
                            'index': i,
                            'file': str(ray_file.name),
                            'iterations': stats.get('iterations', -1),
                            'time': elapsed,
                            'gpu_time': stats.get('gpu_time', 0),
                            'lp_time': stats.get('lp_time', 0),
                            'is_unique': is_new,
                            'rank': actual_rank
                        })
                    else:
                        print(f"  ✗ Ray found but rank insufficient ({actual_rank} < {expected_rank})")
                        log.write(f"  ✗ Rank insufficient: {actual_rank} < {expected_rank}\n")
                        results['statistics']['failed'] += 1
                else:
                    print(f"  ✗ No extremal ray found (time: {elapsed:.1f}s)")
                    log.write(f"  ✗ No ray found or convergence failed\n")
                    results['statistics']['failed'] += 1
                    
            except Exception as e:
                print(f"  ✗ Error during search: {e}")
                log.write(f"  ✗ Error: {e}\n")
                results['statistics']['failed'] += 1
                elapsed = time.time() - start_time
                search_times.append(elapsed)
            
            # Checkpoint save
            if (i + 1) % checkpoint_interval == 0:
                results['last_checkpoint'] = i + 1
                results['avg_time_per_search'] = np.mean(search_times) if search_times else 0
                with open(summary_file, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                print(f"  [Checkpoint saved at search {i+1}]")
                
                # Print performance update
                if len(search_times) > 0:
                    print(f"  [Avg time per search: {np.mean(search_times):.2f}s]")
                    if results['statistics']['total_gpu_time'] > 0:
                        gpu_pct = 100 * results['statistics']['total_gpu_time'] / results['statistics']['total_time']
                        lp_pct = 100 * results['statistics']['total_lp_time'] / results['statistics']['total_time']
                        print(f"  [GPU: {gpu_pct:.1f}%, LP: {lp_pct:.1f}%]")
        
        # Final summary
        log.write(f"\n{'='*60}\n")
        log.write(f"Search completed at {datetime.now()}\n")
        log.write(f"Implementation: {'JAX GPU' if using_jax else 'Phase3'}\n")
        log.write(f"Total searches: {num_searches}\n")
        log.write(f"Extremal rays found: {results['statistics']['total_found']}\n")
        log.write(f"Unique rays: {results['statistics']['unique_found']}\n")
        log.write(f"Failed searches: {results['statistics']['failed']}\n")
        log.write(f"Total time: {results['statistics']['total_time']:.2f}s\n")
        log.write(f"Average time per search: {results['statistics']['total_time']/num_searches:.2f}s\n")
        if results['statistics']['total_gpu_time'] > 0:
            log.write(f"GPU time: {results['statistics']['total_gpu_time']:.2f}s "
                     f"({100*results['statistics']['total_gpu_time']/results['statistics']['total_time']:.1f}%)\n")
            log.write(f"LP time: {results['statistics']['total_lp_time']:.2f}s "
                     f"({100*results['statistics']['total_lp_time']/results['statistics']['total_time']:.1f}%)\n")
    
    # Save final summary
    results['end_time'] = datetime.now().isoformat()
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Print final summary
    print(f"\n{'='*60}")
    print(f"SEARCH COMPLETE - {'JAX GPU' if using_jax else 'Phase3'}")
    print(f"{'='*60}")
    print(f"Total searches: {num_searches}")
    print(f"Extremal rays found: {results['statistics']['total_found']}")
    print(f"Unique rays: {results['statistics']['unique_found']}")
    print(f"Failed searches: {results['statistics']['failed']}")
    print(f"Success rate: {100*results['statistics']['converged']/num_searches:.1f}%")
    print(f"Uniqueness rate: {100*results['statistics']['unique_found']/max(1, results['statistics']['total_found']):.1f}%")
    print(f"Total time: {results['statistics']['total_time']:.2f}s")
    print(f"Avg time/search: {np.mean(search_times) if search_times else 0:.2f}s")
    
    if results['statistics']['total_gpu_time'] > 0:
        print(f"\nPerformance breakdown:")
        print(f"  GPU operations: {100*results['statistics']['total_gpu_time']/results['statistics']['total_time']:.1f}%")
        print(f"  LP solving: {100*results['statistics']['total_lp_time']/results['statistics']['total_time']:.1f}%")
        print(f"  Other: {100*(1 - (results['statistics']['total_gpu_time'] + results['statistics']['total_lp_time'])/results['statistics']['total_time']):.1f}%")
    
    print(f"\nResults saved to: {job_dir}")
    
    return 0

def main():
    parser = argparse.ArgumentParser(description='Find extremal rays with JAX acceleration')
    parser.add_argument('--constraint-file', type=str, required=True,
                       help='Path to .ine constraint file')
    parser.add_argument('--num-searches', type=int, default=100,
                       help='Number of ray searches to perform')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Directory for output files')
    parser.add_argument('--job-id', type=str, default='manual',
                       help='Job ID for tracking (usually SLURM_JOB_ID)')
    parser.add_argument('--checkpoint-interval', type=int, default=10,
                       help='Save checkpoint every N searches')
    parser.add_argument('--use-jax', action='store_true', default=True,
                       help='Use JAX implementation (default: True)')
    parser.add_argument('--no-jax', dest='use_jax', action='store_false',
                       help='Disable JAX, use Phase3 implementation')
    
    args = parser.parse_args()
    
    # Check constraint file exists
    if not os.path.exists(args.constraint_file):
        print(f"ERROR: Constraint file not found: {args.constraint_file}")
        sys.exit(1)
    
    # Run the search
    exit_code = find_extremal_rays_jax(
        args.constraint_file,
        args.num_searches,
        args.output_dir,
        args.job_id,
        args.checkpoint_interval,
        args.use_jax
    )
    
    sys.exit(exit_code)

if __name__ == "__main__":
    main()