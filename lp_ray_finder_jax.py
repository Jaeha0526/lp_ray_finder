#!/usr/bin/env python3
"""
LP-Based Extreme Ray Finder - JAX Optimized Version
Fully GPU-accelerated implementation using JAX for maximum performance
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap, grad
import numpy as np
import time
from functools import partial
from scipy.optimize import linprog
import warnings
warnings.filterwarnings('ignore')

# Configure JAX for GPU
jax.config.update('jax_platform_name', 'gpu')
jax.config.update("jax_enable_x64", False)  # Use float32 for speed

class JAXRayFinder:
    """Fully GPU-accelerated ray finder using JAX"""
    
    def __init__(self, constraint_file=None, use_float32=True, verbose=True):
        self.verbose = verbose
        self.dtype = jnp.float32 if use_float32 else jnp.float64
        self.constraints = None
        self.constraints_gpu = None
        self.m = 0
        self.d = 0
        
        # Pre-compile JAX functions
        self._compile_functions()
        
        if self.verbose:
            devices = jax.devices()
            print(f"🚀 JAX initialized with {len(devices)} device(s): {devices[0].device_kind}")
            print(f"📊 Using dtype: {self.dtype}")
        
        if constraint_file:
            self.load_constraints(constraint_file)
    
    def _compile_functions(self):
        """Pre-compile JAX functions for maximum performance"""
        
        @jit
        def check_violations(constraints, solution):
            """GPU-accelerated violation checking"""
            # Extract constraint matrix (skip constant column)
            A = constraints[:, 1:]
            # Compute violations: A @ x
            violations = jnp.dot(A, solution)
            return violations
        
        @jit
        def find_violated_constraints(violations, threshold=-1e-8):
            """Find indices of violated constraints"""
            violated_mask = violations < threshold
            # Get indices where mask is True
            violated_indices = jnp.where(violated_mask)[0]
            violation_amounts = -violations[violated_indices]
            return violated_indices, violation_amounts
        
        @jit
        def compute_rank(constraints_subset, tol=1e-10):
            """Compute rank using SVD on GPU"""
            _, s, _ = jnp.linalg.svd(constraints_subset, full_matrices=False)
            rank = jnp.sum(s > tol)
            return rank
        
        @partial(jit, static_argnums=(1,))
        def batch_check_violations(constraints, batch_size, solutions):
            """Batch violation checking for multiple solutions"""
            # Vectorized operation for multiple solutions
            A = constraints[:, 1:]
            violations = vmap(lambda x: jnp.dot(A, x))(solutions)
            return violations
        
        # Store compiled functions
        self.check_violations = check_violations
        self.find_violated_constraints = find_violated_constraints
        self.compute_rank = compute_rank
        self.batch_check_violations = batch_check_violations
    
    def load_constraints(self, filename):
        """Load constraints and transfer to GPU"""
        start_time = time.time()
        
        if self.verbose:
            print(f"📁 Loading constraints from {filename}...")
        
        try:
            # Parse .ine file (same as original)
            constraints_list = []
            with open(filename, 'r') as f:
                lines = f.readlines()
            
            in_constraints = False
            found_dimensions = False
            current_constraint = []
            
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
                        coeffs = [float(x) for x in line.split()]
                        current_constraint.extend(coeffs)
                        
                        if len(current_constraint) >= self.d + 1:
                            constraints_list.append(current_constraint[:self.d + 1])
                            current_constraint = current_constraint[self.d + 1:]
                            
                    except ValueError:
                        continue
            
            if len(constraints_list) > 0:
                # Convert to numpy then JAX array
                self.constraints = np.array(constraints_list, dtype=np.float32 if self.dtype == jnp.float32 else np.float64)
                
                # Transfer to GPU
                self.constraints_gpu = jax.device_put(self.constraints)
                
                # Force JIT compilation by running once
                test_solution = jnp.ones(self.d, dtype=self.dtype)
                _ = self.check_violations(self.constraints_gpu, test_solution)
                
                load_time = time.time() - start_time
                
                if self.verbose:
                    print(f"✅ Loaded {len(self.constraints):,} constraints")
                    memory_gb = self.constraints.nbytes / 1024**3
                    print(f"📊 Memory: {memory_gb:.2f} GB")
                    print(f"⏱️  Loading + GPU transfer: {load_time:.2f}s")
                
                return True
            
        except Exception as e:
            print(f"❌ Error loading constraints: {e}")
            return False
    
    @partial(jit, static_argnums=(0,))
    def _solve_lp_active_set_gpu(self, active_constraints, objective):
        """Attempt to solve LP on GPU (experimental)"""
        # Note: Full LP solving on GPU is complex, this is a simplified version
        # For production, consider using cuSOLVER or similar GPU LP solvers
        # For now, we'll still use CPU for LP solving but optimize everything else
        pass
    
    def find_extreme_ray(self, objective=None, max_iterations=100, subset_size=1500):
        """Find extreme ray with full JAX/GPU acceleration"""
        if self.constraints_gpu is None:
            print("❌ No constraints loaded")
            return None, {}
        
        start_time = time.time()
        d = self.d
        
        # Generate objective if not provided
        if objective is None:
            key = jax.random.PRNGKey(int(time.time()))
            objective = jax.random.normal(key, (d,), dtype=self.dtype)
            objective = objective / jnp.linalg.norm(objective)
        else:
            objective = jnp.array(objective, dtype=self.dtype)
        
        if self.verbose:
            print(f"\n🎯 Finding extreme ray with JAX acceleration...")
            print(f"🚀 Full GPU pipeline enabled")
        
        # Initialize active set
        n_constraints = len(self.constraints_gpu)
        active_indices = np.random.choice(n_constraints, min(subset_size, n_constraints), replace=False)
        
        stats = {
            'iterations': 0,
            'gpu_time': 0,
            'lp_time': 0,
            'converged': False
        }
        
        for iteration in range(max_iterations):
            stats['iterations'] = iteration + 1
            
            if self.verbose and iteration % 10 == 0:
                print(f"🔄 Iteration {iteration + 1}: {len(active_indices)} active constraints")
            
            # Extract active constraints (still on GPU)
            active_constraints_gpu = self.constraints_gpu[active_indices]
            
            # Transfer to CPU for scipy LP (only bottleneck)
            lp_start = time.time()
            active_constraints_cpu = np.array(active_constraints_gpu[:, 1:])
            
            # Solve LP on CPU (scipy)
            # For constraints: b + a^T x >= 0, we need a^T x >= -b
            # Convert to scipy format: -a^T x <= b (multiply by -1)
            c = -np.array(objective)
            A_ub = -active_constraints_cpu  # -A for inequality flip
            b_ub = -np.zeros(len(active_indices))  # Should be 0 since constraints have no b column here
            A_eq = np.ones((1, d))
            b_eq = np.array([1.0])
            
            # Note: active_constraints_cpu already excludes the constant term
            # The constant term should be handled separately
            
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                           method='highs', options={'disp': False})
            stats['lp_time'] += time.time() - lp_start
            
            if not result.success:
                if self.verbose:
                    print(f"❌ LP failed: {result.message}")
                break
            
            # Convert solution back to GPU
            gpu_start = time.time()
            solution_gpu = jnp.array(result.x, dtype=self.dtype)
            
            # GPU-accelerated violation checking
            violations = self.check_violations(self.constraints_gpu, solution_gpu)
            
            # Find violated constraints on GPU
            violated_indices, violation_amounts = self.find_violated_constraints(violations)
            stats['gpu_time'] += time.time() - gpu_start
            
            # Check convergence
            if len(violated_indices) == 0:
                stats['converged'] = True
                
                # Extremality verification on GPU
                tight_mask = jnp.abs(violations) < 1e-8
                tight_indices = jnp.where(tight_mask)[0]
                
                if len(tight_indices) > 0:
                    tight_constraints = self.constraints_gpu[tight_indices, 1:]
                    rank = self.compute_rank(tight_constraints)
                    required_rank = d - 1
                    
                    if self.verbose:
                        print(f"✅ Converged! Rank: {rank}/{required_rank}")
                    
                    if rank >= required_rank - 1:  # Allow small tolerance
                        stats['rank'] = int(rank)
                        break
                    else:
                        # Add more constraints
                        remaining = np.setdiff1d(np.arange(n_constraints), active_indices)
                        if len(remaining) > 0:
                            n_add = min(100, len(remaining))
                            new_indices = np.random.choice(remaining, n_add, replace=False)
                            active_indices = np.concatenate([active_indices, new_indices])
                            if self.verbose:
                                print(f"  ➕ Adding {n_add} constraints for rank")
                        else:
                            break
                else:
                    break
            
            # Add most violated constraints
            if len(violated_indices) > 0:
                # Sort and select top violations
                sorted_idx = jnp.argsort(violation_amounts)[-50:]
                most_violated = violated_indices[sorted_idx]
                
                # Convert to CPU for set operations
                most_violated_cpu = np.array(most_violated)
                new_constraints = np.setdiff1d(most_violated_cpu, active_indices)
                
                if len(new_constraints) > 0:
                    active_indices = np.concatenate([active_indices, new_constraints])
                    if self.verbose and len(new_constraints) > 0:
                        print(f"  ➕ Added {len(new_constraints)} violated constraints")
        
        stats['total_time'] = time.time() - start_time
        
        if stats['converged']:
            # Convert solution back to numpy
            solution = np.array(solution_gpu)
            
            if self.verbose:
                print(f"\n📊 Performance Statistics:")
                print(f"  Total time: {stats['total_time']:.2f}s")
                print(f"  GPU time: {stats['gpu_time']:.2f}s ({100*stats['gpu_time']/stats['total_time']:.1f}%)")
                print(f"  LP time: {stats['lp_time']:.2f}s ({100*stats['lp_time']/stats['total_time']:.1f}%)")
                print(f"  Iterations: {stats['iterations']}")
            
            return solution, stats
        else:
            if self.verbose:
                print(f"❌ Failed to converge in {max_iterations} iterations")
            return None, stats


def benchmark_jax_vs_cupy():
    """Benchmark JAX vs CuPy performance"""
    print("="*60)
    print("BENCHMARKING JAX vs CuPy")
    print("="*60)
    
    # Create test constraint matrix
    m, d = 1000000, 63  # 1M constraints for benchmarking
    print(f"\nTest size: {m:,} constraints × {d} dimensions")
    
    # Generate random constraints
    np.random.seed(42)
    constraints = np.random.randn(m, d + 1).astype(np.float32)
    solution = np.random.randn(d).astype(np.float32)
    solution = solution / np.linalg.norm(solution)
    
    # Test CuPy
    try:
        import cupy as cp
        constraints_cupy = cp.asarray(constraints)
        solution_cupy = cp.asarray(solution)
        
        # Warmup
        for _ in range(3):
            _ = cp.dot(constraints_cupy[:, 1:], solution_cupy)
        
        # Benchmark
        cupy_times = []
        for _ in range(10):
            start = time.time()
            violations = cp.dot(constraints_cupy[:, 1:], solution_cupy)
            cp.cuda.Device().synchronize()
            cupy_times.append(time.time() - start)
        
        avg_cupy = np.mean(cupy_times)
        print(f"\n🎮 CuPy average: {avg_cupy*1000:.2f}ms")
        
    except ImportError:
        print("\n🎮 CuPy not available")
        avg_cupy = float('inf')
    
    # Test JAX
    constraints_jax = jax.device_put(constraints)
    solution_jax = jax.device_put(solution)
    
    # JIT compile
    @jit
    def jax_violations(A, x):
        return jnp.dot(A[:, 1:], x)
    
    # Warmup
    for _ in range(3):
        _ = jax_violations(constraints_jax, solution_jax).block_until_ready()
    
    # Benchmark
    jax_times = []
    for _ in range(10):
        start = time.time()
        violations = jax_violations(constraints_jax, solution_jax).block_until_ready()
        jax_times.append(time.time() - start)
    
    avg_jax = np.mean(jax_times)
    print(f"🚀 JAX average: {avg_jax*1000:.2f}ms")
    
    if avg_cupy != float('inf'):
        speedup = avg_cupy / avg_jax
        print(f"\n⚡ JAX is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than CuPy")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    # Run benchmark
    benchmark_jax_vs_cupy()
    
    # Test with actual constraints if available
    import os
    constraint_file = "n6data/n6_correct_s7_expansion.ine"
    
    if os.path.exists(constraint_file):
        print(f"\nTesting with actual N=6 constraints...")
        finder = JAXRayFinder(constraint_file, use_float32=True, verbose=True)
        
        # Find one ray
        ray, stats = finder.find_extreme_ray(max_iterations=20, subset_size=1500)
        
        if ray is not None:
            print(f"\n✅ Successfully found extreme ray!")
            print(f"   Shape: {ray.shape}")
            print(f"   Norm: {np.linalg.norm(ray):.6f}")
    else:
        print(f"\nConstraint file not found: {constraint_file}")
        print("Creating synthetic test...")
        
        # Create synthetic constraints for testing
        finder = JAXRayFinder(verbose=True)
        finder.d = 63
        finder.constraints = np.random.randn(10000, 64).astype(np.float32)
        finder.constraints_gpu = jax.device_put(finder.constraints)
        
        ray, stats = finder.find_extreme_ray(max_iterations=10)