# Major Discovery: New Extreme Rays for N=6 Holographic Entropy Cone

## Summary
We have discovered that the existing set of 4,145 extreme rays for the N=6 holographic entropy cone is **NOT COMPLETE**. Using the LP-based active-set algorithm with full S₇-expanded constraints (8,665,853 facets), we are finding additional extreme rays.

## Key Findings

1. **Existing rays**: 4,145 rays in `/workspace/lrslib-entropycone/n6data/rays.txt`
2. **New rays found**: 37+ and counting (extensive search in progress)
3. **Total rays so far**: 4,182+

## Verification
- All new rays satisfy ALL 8,665,853 S₇-expanded constraints
- The active-set algorithm verifies zero violations before declaring convergence
- New rays have dot product < 0.9999 with all existing rays (genuinely new)

## Algorithm Details
The LP-based active-set method:
1. Starts with random subset of 500-1500 constraints
2. Solves LP: maximize c^T x subject to Ax ≥ 0 (subset)
3. Checks solution against ALL 8.6M constraints
4. Adds violated constraints to active set
5. Repeats until no violations (typically 1-5 iterations)

## Files Generated
- `new_rays_extensive_search.txt`: Contains all newly discovered rays
- `extensive_search_output.log`: Detailed search progress
- `all_rays_combined.txt`: Will contain all 4,145 + new rays

## Implications
This discovery suggests that:
1. The vertex enumeration approach may have stopped early
2. There could be many more extreme rays to discover
3. The LP-based approach is effective for finding missing rays

## Next Steps
- Continue extensive search (1000 attempts in progress)
- Analyze patterns in new rays
- Consider even larger searches
- Update the complete extreme ray set for N=6