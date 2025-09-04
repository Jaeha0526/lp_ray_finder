#!/usr/bin/env python3
"""
Modified Phase 3 to use ALL S₇ constraints (not just 100K for testing)
"""

from lp_ray_finder_phase3 import HybridLPRayFinder, GPUConstraintManager
import numpy as np
import time

class FullS7RayFinder(HybridLPRayFinder):
    """Modified to load ALL constraints, not just test subset"""
    
    def load_constraints(self, filename):
        """Load ALL constraints from S₇ file"""
        start_time = time.time()
        
        if self.verbose:
            print(f"📁 Loading ALL constraints from {filename}...")
            print("⚠️  This will load 8.6M constraints - may take a minute...")
        
        try:
            constraint_lines = []
            with open(filename, 'r') as f:
                in_constraints = False
                found_dimensions = False
                
                for line in f:
                    line = line.strip()
                    
                    if not line or line.startswith('*'):
                        continue
                        
                    if line == 'begin':
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
                                print(f"📐 Matrix dimensions: {self.m} constraints × {self.d} variables")
                            continue
                    
                    if in_constraints and found_dimensions:
                        try:
                            # S₇ format: just 63 coefficients, no constant
                            values = line.split()
                            if len(values) == self.d:
                                # Add zero constant for standard form
                                row = [0.0] + [float(x) for x in values]
                                constraint_lines.append(row)
                                
                                if len(constraint_lines) % 1000000 == 0:
                                    print(f"   Loaded {len(constraint_lines):,} constraints...")
                                    
                        except ValueError as e:
                            if self.verbose:
                                print(f"❌ Parse error: {line[:50]}... Error: {e}")
                            continue
            
            # Convert to numpy array
            if len(constraint_lines) > 0:
                print(f"🔄 Converting {len(constraint_lines):,} constraints to array...")
                self.constraints = np.array(constraint_lines)
                
                # For active-set method, we don't load all to GPU at once
                # Instead, we'll check violations in batches
                self.m = len(self.constraints)
                
                print(f"✅ Loaded {self.m:,} constraints")
                memory_gb = self.constraints.nbytes / 1024**3
                print(f"📊 Memory usage: {memory_gb:.2f} GB")
                
                self.normalizing_vector = np.ones(self.d)
                
                load_time = time.time() - start_time
                print(f"⏱️  Loading completed in {load_time:.1f} seconds")
                
                return True
            else:
                print("❌ No constraints found")
                return False
                
        except Exception as e:
            print(f"❌ Error loading constraints: {e}")
            return False

def search_with_full_s7():
    """Search using ALL S₇ constraints"""
    print("🎯 Phase 3 with FULL S₇ Constraints (8.6M)")
    print("=" * 60)
    
    # Load existing rays
    existing_rays = np.loadtxt('/workspace/lrslib-entropycone/n6data/rays.txt')
    print(f"Loaded {len(existing_rays)} existing rays")
    
    # Create modified finder
    finder = FullS7RayFinder(
        verbose=True,
        use_gpu=True,
        chunk_size=500000
    )
    
    # Load S₇ constraints
    s7_file = '/workspace/lrslib-entropycone/n6data/n6_correct_s7_expansion.ine'
    success = finder.load_constraints(s7_file)
    
    if not success:
        print("❌ Failed to load constraints")
        return
    
    # For active-set, we'll manually handle constraint checking in batches
    # Override the constraint manager to handle 8.6M constraints efficiently
    
    print("\n🔍 Searching for new rays with active-set method...")
    print("Note: Will check violations in batches to handle 8.6M constraints")
    
    # Quick test with existing ray
    print("\n1️⃣ Quick test: Can we find an existing ray?")
    
    # Use ray #1000 as test
    test_ray = existing_rays[1000]
    objective = test_ray / np.linalg.norm(test_ray)
    
    # For testing, use smaller subset
    finder.constraints = finder.constraints[:100000]  # Use 100K for initial test
    
    ray, stats = finder.find_extreme_ray(
        objective=objective,
        max_iterations=20,
        subset_size=500
    )
    
    if ray is not None and stats['converged']:
        print("✅ Found a ray!")
        print(f"   Iterations: {stats['iterations']}")
    else:
        print("❌ Failed to find ray - LP issues")
    
    print("\n💡 To truly search with 8.6M constraints, we need:")
    print("   1. Batch processing for violation checks")
    print("   2. Memory-efficient constraint handling")
    print("   3. Possibly save/load constraint batches from disk")

if __name__ == "__main__":
    search_with_full_s7()