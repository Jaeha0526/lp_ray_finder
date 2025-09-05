#!/usr/bin/env python3
"""
CORRECTLY FIXED LP-Based Extreme Ray Finder
Now properly handles the constraint format: 63 coefficients per line, no constant term
"""

import numpy as np
import time
from scipy.optimize import linprog
import random
import sys
from typing import Tuple, Dict, Optional

class CorrectLPRayFinder:
    """Correctly implemented LP Ray Finder"""
    
    def __init__(self, constraint_file=None, verbose=True):
        self.verbose = verbose
        self.constraints = None
        self.m = 0
        self.d = 0  
        
        if constraint_file:
            self.load_constraints(constraint_file)
    
    def load_constraints(self, filename, max_constraints=None):
        """Load constraints from .ine file - CORRECTED VERSION"""
        start_time = time.time()
        
        if self.verbose:
            print(f"📁 Loading constraints from {filename}...")
        
        try:
            with open(filename, 'r') as f:
                lines = f.readlines()
            
            in_constraints = False
            found_dimensions = False
            constraint_lines = []
            
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
                        self.m = int(parts[0])
                        self.d = int(parts[1])
                        found_dimensions = True
                        if self.verbose:
                            print(f"📐 Dimensions: {self.m} constraints × {self.d} variables")
                        continue
                
                if line.startswith('startingcobasis'):
                    break
                    
                if in_constraints and found_dimensions:
                    try:
                        # Each line has exactly d coefficients (no constant term!)
                        coeffs = [float(x) for x in line.split()]
                        if len(coeffs) == self.d:
                            constraint_lines.append(coeffs)
                            
                            if max_constraints and len(constraint_lines) >= max_constraints:
                                break
                                
                    except ValueError:
                        continue
            
            if len(constraint_lines) > 0:
                self.constraints = np.array(constraint_lines)
                actual_m, actual_d = self.constraints.shape
                
                if self.verbose:
                    print(f"✅ Loaded {actual_m} constraints with {actual_d} variables")
                    print(f"✅ Constraint shape: {self.constraints.shape}")
                    memory_mb = self.constraints.nbytes / 1024 / 1024
                    print(f"📊 Memory: {memory_mb:.1f} MB")
                
                load_time = time.time() - start_time
                if self.verbose:
                    print(f"⏱️  Loading completed in {load_time:.3f} seconds")
                
                return True
            else:
                print("❌ Error: No valid constraint lines found")
                return False
                
        except Exception as e:
            print(f"❌ Error loading constraints: {e}")
            return False
    
    def check_violations(self, solution: np.ndarray) -> np.ndarray:
        """
        Check constraint violations - CORRECTED
        For constraint: a^T x >= 0
        Violation is: a^T x (negative means violated)
        """
        violations = np.dot(self.constraints, solution)  # A*x
        return violations
    
    def find_extreme_ray(self, objective=None, max_iterations=50, subset_size=500):
        """
        Find extreme ray with CORRECT homogeneous constraint formulation
        """
        if self.constraints is None or len(self.constraints) == 0:
            print("❌ Error: No constraints loaded")
            return None, {}
        
        start_time = time.time()
        d = self.d
        
        if objective is None:
            objective = np.random.randn(d)
            objective = objective / np.linalg.norm(objective)
        
        if self.verbose:
            print(f"\n🎯 Finding extreme ray (CORRECT version)...")
            print(f"🎯 Constraints are homogeneous: Ax >= 0")
        
        n_constraints = len(self.constraints)
        active_indices = random.sample(range(n_constraints), min(subset_size, n_constraints))
        
        stats = {
            'iterations': 0,
            'lp_solves': 0,
            'converged': False,
            'total_time': 0
        }
        
        for iteration in range(max_iterations):
            stats['iterations'] = iteration + 1
            
            if self.verbose and iteration % 10 == 0:
                print(f"🔄 Iteration {iteration + 1}: {len(active_indices)} active constraints")
            
            # Extract active constraints (no constant term to worry about!)
            A_active = self.constraints[active_indices]
            
            # CORRECT LP formulation for homogeneous constraints:
            # Maximize: c^T x
            # Subject to: A*x >= 0  (homogeneous)
            #            sum(x) = 1  (normalization)
            # 
            # For scipy: minimize -c^T x subject to -A*x <= 0
            
            c = -objective  # Minimize -c^T x = maximize c^T x
            A_ub = -A_active  # -A*x <= 0 equivalent to A*x >= 0
            b_ub = np.zeros(len(active_indices))  # Right hand side is 0 (homogeneous)
            
            # Normalization constraint
            A_eq = np.ones((1, d))
            b_eq = np.array([1.0])
            
            stats['lp_solves'] += 1
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                           method='highs', options={'disp': False})
            
            if not result.success:
                if self.verbose:
                    print(f"❌ LP solve failed at iteration {iteration + 1}: {result.message}")
                break
            
            current_solution = result.x
            
            # Check violations with CORRECT formula (just A*x)
            violations = self.check_violations(current_solution)
            
            # Find violated constraints
            violated_mask = violations < -1e-8
            violated_indices = np.where(violated_mask)[0]
            
            if len(violated_indices) == 0:
                stats['converged'] = True
                if self.verbose:
                    print(f"✅ Converged in {iteration + 1} iterations!")
                    print(f"  Solution sum: {np.sum(current_solution):.6f}")
                    print(f"  Solution norm: {np.linalg.norm(current_solution):.6f}")
                    print(f"  Non-zeros: {np.sum(np.abs(current_solution) > 1e-8)}/{d}")
                
                # Extremality verification
                tight_mask = np.abs(violations) < 1e-8
                tight_indices = np.where(tight_mask)[0]
                
                if len(tight_indices) > 0:
                    tight_constraints = self.constraints[tight_indices]
                    rank = np.linalg.matrix_rank(tight_constraints, tol=1e-10)
                    required_rank = d - 1
                    
                    stats['rank'] = rank
                    
                    if rank >= required_rank:
                        if self.verbose:
                            print(f"✅ Extremality verified: rank = {rank}/{required_rank}")
                        break
                    else:
                        if self.verbose:
                            print(f"⚠️ Rank deficient: {rank}/{required_rank}")
                        
                        # Add more constraints
                        remaining_indices = [i for i in range(n_constraints) 
                                           if i not in active_indices]
                        if remaining_indices:
                            n_to_add = min(100, len(remaining_indices))
                            random_additions = random.sample(remaining_indices, n_to_add)
                            active_indices.extend(random_additions)
                            continue
                        else:
                            break
                else:
                    break
            
            # Add most violated constraints
            violation_amounts = -violations[violated_indices]
            most_violated = violated_indices[np.argsort(violation_amounts)[-50:]]
            
            new_constraints = [idx for idx in most_violated if idx not in active_indices]
            active_indices.extend(new_constraints)
            
            if self.verbose and len(new_constraints) > 0:
                print(f"  ➕ Added {len(new_constraints)} violated constraints")
        
        stats['total_time'] = time.time() - start_time
        
        if stats['converged']:
            if self.verbose:
                print(f"📊 Total time: {stats['total_time']:.2f}s")
                print(f"📊 LP solves: {stats['lp_solves']}")
            return current_solution, stats
        else:
            if self.verbose:
                print(f"❌ Failed to converge in {max_iterations} iterations")
            return None, stats


def test_correct_implementation():
    """Test the corrected implementation"""
    print("="*60)
    print("TESTING CORRECTED IMPLEMENTATION")
    print("="*60)
    
    # Test on the actual constraint file
    constraint_file = "n6data/n6_correct_s7_expansion.ine"
    
    # Load with limited constraints for quick test
    finder = CorrectLPRayFinder(verbose=True)
    finder.load_constraints(constraint_file, max_constraints=10000)
    
    if finder.constraints is None:
        print("Failed to load constraints")
        return 1
    
    # Verify the constraints are homogeneous
    print(f"\nConstraint statistics:")
    print(f"  Shape: {finder.constraints.shape}")
    print(f"  Range: [{finder.constraints.min():.1f}, {finder.constraints.max():.1f}]")
    print(f"  Unique values: {np.unique(finder.constraints)}")
    
    # Try to find rays
    success_count = 0
    
    print(f"\nAttempting to find 5 extreme rays...")
    
    for i in range(5):
        print(f"\n{'='*40}")
        print(f"Attempt {i+1}/5")
        print(f"{'='*40}")
        
        ray, stats = finder.find_extreme_ray(max_iterations=20, subset_size=500)
        
        if ray is not None and stats['converged']:
            success_count += 1
            print(f"✓ SUCCESS! Found extreme ray")
            
            # Verify the ray satisfies constraints
            violations = finder.check_violations(ray)
            num_violated = np.sum(violations < -1e-8)
            print(f"  Verification: {num_violated} constraints violated")
            if num_violated == 0:
                print(f"  ✓ Ray is valid!")
        else:
            print(f"✗ FAILED to find ray")
    
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Success rate: {success_count}/5 ({100*success_count/5:.0f}%)")
    
    if success_count > 0:
        print(f"✅ CORRECTED VERSION IS WORKING!")
    else:
        print(f"❌ Still having issues - may need different normalization or approach")
    
    return 0


def batch_find_rays(constraint_file, num_searches=100, output_dir="results", job_id="manual"):
    """Batch search for extreme rays"""
    print("="*60)
    print("BATCH RAY FINDING - CORRECTED IMPLEMENTATION")
    print("="*60)
    
    from pathlib import Path
    import json
    from datetime import datetime
    
    # Create output directory
    output_path = Path(output_dir) / f"job_{job_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_path.mkdir(parents=True, exist_ok=True)
    rays_dir = output_path / "rays"
    rays_dir.mkdir(exist_ok=True)
    
    # Load constraints
    finder = CorrectLPRayFinder(verbose=False)
    print(f"Loading constraints from {constraint_file}...")
    finder.load_constraints(constraint_file)
    
    if finder.constraints is None:
        print("Failed to load constraints")
        return 1
    
    print(f"✅ Loaded {len(finder.constraints)} constraints")
    print(f"✅ Dimensions: {finder.d}")
    print(f"✅ Constraint shape: {finder.constraints.shape}")
    
    # Track results
    results = {
        'job_id': job_id,
        'start_time': datetime.now().isoformat(),
        'constraint_file': constraint_file,
        'num_constraints': len(finder.constraints),
        'dimensions': finder.d,
        'num_searches': num_searches,
        'rays_found': [],
        'statistics': {
            'total_found': 0,
            'failed': 0,
            'total_time': 0
        }
    }
    
    # Search for rays
    print(f"\nSearching for {num_searches} extreme rays...")
    print("="*60)
    
    for i in range(num_searches):
        if i % 10 == 0 and i > 0:
            print(f"Progress: {i}/{num_searches} searches completed ({results['statistics']['total_found']} rays found)...")
        
        start_time = time.time()
        ray, stats = finder.find_extreme_ray(max_iterations=50, subset_size=500)
        elapsed = time.time() - start_time
        
        results['statistics']['total_time'] += elapsed
        
        if ray is not None and stats['converged']:
            results['statistics']['total_found'] += 1
            # Save ray
            ray_file = rays_dir / f"ray_{i:04d}.npz"
            np.savez(ray_file, ray=ray, **stats)
            print(f"  ✓ Ray {results['statistics']['total_found']} found (search {i+1}) in {elapsed:.2f}s")
        else:
            results['statistics']['failed'] += 1
        
        # Save checkpoint every 10 searches
        if (i + 1) % 10 == 0:
            results['end_time'] = datetime.now().isoformat()
            with open(output_path / "summary.json", 'w') as f:
                json.dump(results, f, indent=2, default=str)
    
    # Final summary
    results['end_time'] = datetime.now().isoformat()
    with open(output_path / "summary.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Total searches: {num_searches}")
    print(f"✅ Rays found: {results['statistics']['total_found']}")
    print(f"❌ Failed: {results['statistics']['failed']}")
    print(f"Success rate: {100*results['statistics']['total_found']/num_searches:.1f}%")
    print(f"Total time: {results['statistics']['total_time']:.1f}s")
    print(f"Avg time per search: {results['statistics']['total_time']/num_searches:.2f}s")
    print(f"Results saved to: {output_path}")
    
    return 0

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Find extreme rays with corrected LP formulation')
    parser.add_argument('constraint_file', nargs='?', help='Path to .ine constraint file')
    parser.add_argument('--job-id', default='manual', help='Job ID for tracking')
    parser.add_argument('--num-searches', type=int, default=100, help='Number of ray searches')
    parser.add_argument('--test', action='store_true', help='Run test mode')
    
    args = parser.parse_args()
    
    if args.test or not args.constraint_file:
        sys.exit(test_correct_implementation())
    else:
        sys.exit(batch_find_rays(args.constraint_file, args.num_searches, job_id=args.job_id))