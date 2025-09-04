#!/usr/bin/env python3
"""
Convert rays to their minimal integer representation
"""

import numpy as np
from fractions import Fraction
import math

def to_integer_ray(ray, tolerance=1e-9):
    """Convert a ray to minimal integer coordinates"""
    # Normalize the ray
    ray_norm = ray / np.linalg.norm(ray)
    
    # Convert to fractions to find exact ratios
    fractions = []
    for val in ray_norm:
        # Find the best rational approximation
        frac = Fraction(val).limit_denominator(10000)
        fractions.append(frac)
    
    # Find the LCM of all denominators
    denominators = [f.denominator for f in fractions]
    lcm = denominators[0]
    for d in denominators[1:]:
        lcm = lcm * d // math.gcd(lcm, d)
    
    # Scale by LCM to get integers
    integer_ray = []
    for f in fractions:
        integer_ray.append(int(f * lcm))
    
    # Find GCD and reduce
    gcd = math.gcd(*[abs(x) for x in integer_ray if x != 0])
    if gcd > 1:
        integer_ray = [x // gcd for x in integer_ray]
    
    return integer_ray

def analyze_rays():
    """Analyze and convert rays to integer form"""
    print("🔢 Converting Rays to Integer Form")
    print("="*60)
    
    # Load the unique rays
    rays = np.loadtxt('signature_unique_rays.txt')
    print(f"Loaded {len(rays)} unique orbit representatives")
    
    # Convert each ray
    integer_rays = []
    for i, ray in enumerate(rays):
        int_ray = to_integer_ray(ray)
        integer_rays.append(int_ray)
        
        print(f"\n📌 Ray {i+1}:")
        print(f"   First 10 float values: {ray[:10]}")
        print(f"   First 10 integer values: {int_ray[:10]}")
        
        # Check the pattern
        unique_vals = sorted(set(int_ray))
        print(f"   Unique integer values: {unique_vals}")
        print(f"   Max value: {max(int_ray)}")
    
    # Save integer rays
    with open('integer_rays.txt', 'w') as f:
        for ray in integer_rays:
            f.write(' '.join(map(str, ray)) + '\n')
    
    print(f"\n💾 Saved integer rays to: integer_rays.txt")
    
    # Also check some from extended search
    if np.DataSource().exists('extended_search_new_rays.txt'):
        print("\n📊 Checking extended search rays...")
        try:
            extended = np.loadtxt('extended_search_new_rays.txt')
            if extended.ndim == 1:
                extended = extended.reshape(1, -1)
            
            # Check first few
            for i in range(min(5, len(extended))):
                int_ray = to_integer_ray(extended[i])
                print(f"\nExtended ray {i+1}:")
                print(f"   Integer form (first 10): {int_ray[:10]}")
                print(f"   Max value: {max(int_ray)}")
        except:
            pass

if __name__ == "__main__":
    analyze_rays()