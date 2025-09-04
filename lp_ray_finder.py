#!/usr/bin/env python3
"""
LP-Based Extreme Ray Finder for Large Convex Cones

This implements the active-set LP algorithm to find extreme rays
of convex cones without full vertex enumeration.
"""

import numpy as np
import time
from scipy.optimize import linprog
import random
import os
import sys

class LPRayFinder:
    def __init__(self, constraint_file=None, verbose=True):
        """
        Initialize LP Ray Finder
        
        Args:
            constraint_file: Path to .ine file with constraints
            verbose: Print progress messages
        """
        self.verbose = verbose
        self.constraints = None  # V matrix (m x d)
        self.m = 0  # number of constraints  
        self.d = 0  # number of dimensions
        self.normalizing_vector = None  # h vector
        
        if constraint_file:
            self.load_constraints(constraint_file)
    
    def load_constraints(self, filename):
        """Load constraints from .ine file format"""
        if self.verbose:
            print(f"Loading constraints from {filename}...")
        
        try:
            with open(filename, 'r') as f:
                lines = f.readlines()
            
            # Parse .ine format
            in_constraints = False
            found_dimensions = False
            constraint_lines = []
            current_constraint = []
            line_idx = 0
            
            while line_idx < len(lines):
                line = lines[line_idx].strip()
                line_idx += 1
                
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
                        self.m = int(parts[0])  # constraints
                        self.d = int(parts[1])  # dimensions  
                        found_dimensions = True
                        if self.verbose:
                            print(f"Matrix dimensions: {self.m} constraints × {self.d} variables")
                        continue
                
                if line.strip().startswith('startingcobasis'):
                    break
                    
                if in_constraints and found_dimensions:
                    # Parse multi-line constraint format
                    try:
                        # Collect numbers from current line
                        coeffs = [float(x) for x in line.split()]
                        current_constraint.extend(coeffs)
                        
                        # Check if we have enough coefficients for a complete constraint
                        if len(current_constraint) >= self.d + 1:
                            # We have a complete constraint
                            constraint_lines.append(current_constraint[:self.d + 1])
                            
                            # Debug first few constraints
                            if len(constraint_lines) <= 3 and self.verbose:
                                print(f"✅ Constraint {len(constraint_lines)}: {constraint_lines[-1][:5]}... (first 5 elements)")
                            
                            # Reset for next constraint
                            current_constraint = current_constraint[self.d + 1:]
                            
                            # Limit for testing
                            if len(constraint_lines) >= 1000:
                                if self.verbose:
                                    print(f"Limiting to {len(constraint_lines)} constraints for testing")
                                break
                        
                    except ValueError as e:
                        if self.verbose and len(constraint_lines) < 3:
                            print(f"❌ Parse error: {line[:50]}... Error: {e}")
                        continue
            
            # Convert to numpy array
            if len(constraint_lines) > 0:
                self.constraints = np.array(constraint_lines)
                actual_m, actual_d_plus_1 = self.constraints.shape
            else:
                print("Error: No valid constraint lines found")
                return False
            
            if self.verbose:
                print(f"Loaded {actual_m} constraints with {actual_d_plus_1-1} variables")
                print(f"Constraint matrix shape: {self.constraints.shape}")
            
            # Set up normalizing vector (use sum of absolute values)
            self.normalizing_vector = np.ones(self.d)  # Simple choice for testing
            
            return True
            
        except Exception as e:
            print(f"Error loading constraints: {e}")
            return False
    
    def find_extreme_ray(self, objective=None, max_iterations=50, subset_size=100):
        """
        Find one extreme ray using active-set LP method
        
        Args:
            objective: Objective vector c (random if None)
            max_iterations: Max iterations for active-set
            subset_size: Initial constraint subset size
            
        Returns:
            ray: Extreme ray vector (or None if failed)
            stats: Dictionary with solving statistics
        """
        if self.constraints is None or len(self.constraints) == 0:
            print("Error: No constraints loaded")
            return None, {}
        
        start_time = time.time()
        stats = {
            'iterations': 0,
            'lp_solves': 0,
            'violation_checks': 0,
            'total_time': 0,
            'converged': False
        }
        
        # Set up problem
        d = self.d
        
        # Random objective if not provided
        if objective is None:
            objective = np.random.randn(d)
            objective = objective / np.linalg.norm(objective)  # normalize
        
        if self.verbose:
            print(f"\\nFinding extreme ray with active-set LP method...")
            print(f"Objective vector: {objective[:5]}... (showing first 5 elements)")
        
        # Start with random subset of constraints
        n_constraints = len(self.constraints)
        active_indices = random.sample(range(n_constraints), min(subset_size, n_constraints))
        
        for iteration in range(max_iterations):
            stats['iterations'] = iteration + 1
            
            if self.verbose and iteration % 5 == 0:
                print(f"Iteration {iteration + 1}: {len(active_indices)} active constraints")
            
            # Extract active constraints (skip constant term for now - simplified)
            active_constraints = self.constraints[active_indices, 1:]  # Skip constant column
            
            # Solve LP: max c^T w  s.t.  A w >= 0, h^T w = 1
            # Convert to scipy format: min -c^T w  s.t.  -A w <= 0, h^T w = 1
            c = -objective  # minimize negative = maximize positive
            A_ub = -active_constraints  # -A w <= 0  =>  A w >= 0
            b_ub = np.zeros(len(active_indices))
            A_eq = self.normalizing_vector.reshape(1, -1)  # h^T w = 1
            b_eq = np.array([1.0])
            
            # Solve LP
            stats['lp_solves'] += 1
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                           method='highs', options={'disp': False})
            
            if not result.success:
                if self.verbose:
                    print(f"LP solve failed at iteration {iteration + 1}: {result.message}")
                break
            
            current_solution = result.x
            
            # Check violations across ALL constraints
            stats['violation_checks'] += 1
            all_constraints = self.constraints[:, 1:]  # Skip constant column
            violations = np.dot(all_constraints, current_solution)  # Should be >= 0
            
            # Find most violated constraints
            violated_mask = violations < -1e-8  # tolerance for numerical errors
            violated_indices = np.where(violated_mask)[0]
            
            if len(violated_indices) == 0:
                # No violations - we found an extreme ray!
                stats['converged'] = True
                if self.verbose:
                    print(f"\\nConverged in {iteration + 1} iterations!")
                    print(f"Found extreme ray: {current_solution[:5]}... (showing first 5 elements)")
                break
            
            # Add most violated constraints to active set
            violation_amounts = -violations[violated_indices]  # negative violations
            most_violated = violated_indices[np.argsort(violation_amounts)[-10:]]  # top 10
            
            new_constraints = [idx for idx in most_violated if idx not in active_indices]
            active_indices.extend(new_constraints)
            
            if self.verbose and len(new_constraints) > 0:
                print(f"  Added {len(new_constraints)} violated constraints")
        
        stats['total_time'] = time.time() - start_time
        
        if stats['converged']:
            return current_solution, stats
        else:
            if self.verbose:
                print(f"Failed to converge in {max_iterations} iterations")
            return None, stats

def test_small_example():
    """Test with a simple 2D example"""
    print("=== TESTING WITH SIMPLE 2D EXAMPLE ===")
    
    # Create simple 2D cone: x >= 0, y >= 0, x + y >= 1
    constraints = np.array([
        [0, 1, 0],    # x >= 0
        [0, 0, 1],    # y >= 0  
        [1, 1, 1],    # x + y >= 1
    ])
    
    finder = LPRayFinder()
    finder.constraints = constraints
    finder.d = 2
    finder.normalizing_vector = np.array([1, 1])  # x + y = 1 normalization
    
    ray, stats = finder.find_extreme_ray()
    
    print(f"\\nResults:")
    print(f"Ray found: {ray}")
    print(f"Stats: {stats}")
    
    return ray is not None

def test_real_ine_file():
    """Test with a real .ine file from lrslib"""
    print("\\n=== TESTING WITH REAL .INE FILE ===")
    
    # Try a small real example
    test_file = "../ine/test/truss2.ine"
    if os.path.exists(test_file):
        print(f"Testing with {test_file}")
        finder = LPRayFinder(test_file)
        
        if finder.constraints is not None:
            print(f"✅ Successfully loaded {finder.constraints.shape[0]} constraints")
            ray, stats = finder.find_extreme_ray()
            
            print(f"\\nReal File Test Results:")
            print(f"Converged: {stats['converged']}")
            print(f"Iterations: {stats['iterations']}")
            print(f"Total time: {stats['total_time']:.4f} seconds")
            
            if ray is not None:
                print(f"Ray found: {ray}")
                print("SUCCESS: LP method works on real .ine files!")
                return True
            else:
                print("No ray found")
                return False
        else:
            print("Failed to load constraints")
            return False
    else:
        print(f"Test file not found: {test_file}")
        return False

def main():
    """Main test function"""
    print("LP-Based Ray Finder - Proof of Concept")
    print("=" * 50)
    
    # Test 1: Simple example
    if not test_small_example():
        print("Simple test failed!")
        return
    
    # Test 2: Real .ine file
    if not test_real_ine_file():
        print("Real file test failed!")
        return
    
    # Test 3: Try with actual N=6 data if available
    n6_file = "../n6data/n6_restart_startingcobasis.ine"
    if os.path.exists(n6_file):
        print("\\n" + "=" * 50)
        print("=== TESTING WITH N=6 DATA (LIMITED) ===")
        
        finder = LPRayFinder(n6_file)
        if finder.constraints is not None:
            ray, stats = finder.find_extreme_ray()
            
            print(f"\\nN=6 Test Results:")
            print(f"Converged: {stats['converged']}")
            print(f"Iterations: {stats['iterations']}")
            print(f"Total time: {stats['total_time']:.2f} seconds")
            
            if ray is not None:
                print(f"Ray found: {ray[:5]}... (first 5 elements)")
                print("SUCCESS: LP method found an extreme ray!")
            else:
                print("No ray found in this test")
    else:
        print(f"\\nN=6 file not found at {n6_file}")
    
    print("\\n" + "=" * 50)
    print("Proof of concept complete!")

if __name__ == "__main__":
    main()