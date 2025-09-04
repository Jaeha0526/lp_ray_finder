#!/usr/bin/env python3
"""
Extract exact integer coordinate expressions for the 16 new orbit representatives
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
    
    # Find GCD and reduce to minimal form
    nonzero_vals = [abs(x) for x in integer_ray if x != 0]
    if nonzero_vals:
        gcd = math.gcd(*nonzero_vals)
        if gcd > 1:
            integer_ray = [x // gcd for x in integer_ray]
    
    return integer_ray

def format_vector_compact(vec):
    """Format vector in compact notation"""
    # Group consecutive identical values
    result = []
    i = 0
    while i < len(vec):
        val = vec[i]
        count = 1
        while i + count < len(vec) and vec[i + count] == val:
            count += 1
        
        if count == 1:
            result.append(str(val))
        else:
            result.append(f"{val}×{count}")
        i += count
    
    return "[" + ", ".join(result) + "]"

def analyze_pattern(vec):
    """Analyze the pattern in the vector"""
    unique_vals = sorted(set(vec))
    counts = {val: vec.count(val) for val in unique_vals}
    
    analysis = {
        'unique_values': unique_vals,
        'value_counts': counts,
        'range': max(vec) - min(vec),
        'gcd': math.gcd(*[abs(x) for x in vec if x != 0]) if any(x != 0 for x in vec) else 0
    }
    
    return analysis

def main():
    print("🔢 Integer Coordinate Expressions for 16 New Orbit Representatives")
    print("="*80)
    
    # Load the 16 unique rays
    rays = np.loadtxt('extended_unique_rays.txt')
    
    all_integer_rays = []
    
    for i, ray in enumerate(rays):
        print(f"\n{'='*60}")
        print(f"🎯 NEW ORBIT REPRESENTATIVE #{i+1}")
        print('='*60)
        
        # Convert to integer coordinates
        int_ray = to_integer_ray(ray)
        all_integer_rays.append(int_ray)
        
        # Analyze the pattern
        pattern = analyze_pattern(int_ray)
        
        print(f"📐 Full Integer Vector (63 dimensions):")
        print(f"   {int_ray}")
        
        print(f"\n📊 Pattern Analysis:")
        print(f"   Unique values: {pattern['unique_values']}")
        print(f"   Value counts: {pattern['value_counts']}")
        print(f"   Range: {pattern['range']} (max - min)")
        print(f"   GCD: {pattern['gcd']}")
        
        print(f"\n🎨 Compact Representation:")
        compact = format_vector_compact(int_ray)
        print(f"   {compact}")
        
        # Show structure by position groups (entropy cone structure)
        print(f"\n🏗️  Entropy Cone Structure:")
        print(f"   S(∅): {int_ray[0]}")
        print(f"   S(1-7): {int_ray[1:8]}")
        print(f"   S(pairs): {int_ray[8:29]}")  # 21 pairs
        print(f"   S(triples): {int_ray[29:64]}")  # 35 triples (but we use 63 total)
        
        # Check for simple ratios
        if len(pattern['unique_values']) <= 5:
            print(f"\n✨ Simple Integer Pattern Detected!")
            ratios = []
            base = min([x for x in pattern['unique_values'] if x > 0]) if any(x > 0 for x in pattern['unique_values']) else 1
            for val in pattern['unique_values']:
                if val == 0:
                    ratios.append("0")
                else:
                    ratio = val // base if val % base == 0 else f"{val}/{base}"
                    ratios.append(str(ratio))
            print(f"   Base unit: {base}")
            print(f"   Ratios: {ratios}")
    
    # Save all integer rays
    print(f"\n\n💾 Saving Integer Coordinates...")
    with open('new_16_orbits_integer_coordinates.txt', 'w') as f:
        f.write("# 16 New Orbit Representatives - Integer Coordinates\n")
        f.write("# Each line is one 63-dimensional extreme ray in minimal integer form\n")
        f.write("# Format: ray_number: [coordinates]\n\n")
        
        for i, int_ray in enumerate(all_integer_rays):
            f.write(f"# Orbit {i+1}:\n")
            f.write(" ".join(map(str, int_ray)) + "\n\n")
    
    print("✅ Integer coordinates saved to: new_16_orbits_integer_coordinates.txt")
    
    # Summary statistics
    print(f"\n📈 SUMMARY STATISTICS")
    print("="*50)
    
    max_values = [max(ray) for ray in all_integer_rays]
    min_values = [min(ray) for ray in all_integer_rays]
    unique_counts = [len(set(ray)) for ray in all_integer_rays]
    
    print(f"Maximum coordinate values: {max_values}")
    print(f"Minimum coordinate values: {min_values}")
    print(f"Unique value counts: {unique_counts}")
    
    simple_patterns = sum(1 for count in unique_counts if count <= 5)
    complex_patterns = len(unique_counts) - simple_patterns
    
    print(f"\n🎯 Pattern Classification:")
    print(f"   Simple patterns (≤5 unique values): {simple_patterns}")
    print(f"   Complex patterns (>5 unique values): {complex_patterns}")

if __name__ == "__main__":
    main()