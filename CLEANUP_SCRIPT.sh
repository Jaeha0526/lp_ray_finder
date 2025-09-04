#!/bin/bash
# Cleanup script for lp_ray_finder directory
# Removes test files, analysis scripts, and temporary files

echo "🧹 Cleaning up lp_ray_finder directory..."

# Create backup directory for important files
mkdir -p ../lp_ray_finder_backup

# Core algorithm files to KEEP
CORE_FILES=(
    "lp_ray_finder_phase3.py"
    "lp_ray_finder_phase2.py" 
    "lp_ray_finder.py"
    "README.md"
    "ALGORITHM_DOCUMENTATION.md"
)

# Discovery results to KEEP
RESULT_FILES=(
    "all_unique_rays_final.txt"
    "all_unique_rays_integer.txt"
    "FINAL_DISCOVERY_REPORT.md"
    "VERIFIED_FINAL_RESULTS.md"
    "N6_NEW_RAYS_DISCOVERY_SUMMARY.md"
    "DISCOVERY_SUMMARY.md"
)

echo "📋 Files to remove (test/analysis/temporary):"

# List files to be removed
find . -name "test_*.py" -o \
       -name "demo_*.py" -o \
       -name "check_*.py" -o \
       -name "analyze_*.py" -o \
       -name "monitor_*.py" -o \
       -name "validate_*.py" -o \
       -name "verify_*.py" -o \
       -name "debug_*.py" -o \
       -name "*_test.py" -o \
       -name "*_demo.py" -o \
       -name "*_check.py" -o \
       -name "simple_*.py" -o \
       -name "thorough_*.py" -o \
       -name "gpu_*.py" -o \
       -name "launch_*.py" -o \
       -name "wake_*.py" -o \
       -name "run_*.py" | head -20

echo ""
echo "🚫 Files to REMOVE (will be moved to backup):"
echo "   - All test_*.py files"
echo "   - All demo_*.py files"  
echo "   - All check_*.py files"
echo "   - All analyze_*.py files"
echo "   - All monitor_*.py files"
echo "   - All temporary .txt search results"
echo "   - All .log files"
echo ""

echo "✅ Files to KEEP:"
printf "   - %s\n" "${CORE_FILES[@]}"
printf "   - %s\n" "${RESULT_FILES[@]}"
echo "   - n6_new_rays_discovery/ directory"
echo ""

read -p "Continue with cleanup? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🧹 Starting cleanup..."
    
    # Move test/analysis files to backup
    find . -maxdepth 1 \( \
        -name "test_*.py" -o \
        -name "demo_*.py" -o \
        -name "check_*.py" -o \
        -name "analyze_*.py" -o \
        -name "monitor_*.py" -o \
        -name "validate_*.py" -o \
        -name "verify_*.py" -o \
        -name "debug_*.py" -o \
        -name "*_test.py" -o \
        -name "*_demo.py" -o \
        -name "*_check.py" -o \
        -name "simple_*.py" -o \
        -name "thorough_*.py" -o \
        -name "gpu_*.py" -o \
        -name "launch_*.py" -o \
        -name "wake_*.py" -o \
        -name "run_*.py" -o \
        -name "*search*.py" -o \
        -name "*search*.txt" -o \
        -name "*.log" \
    \) -exec mv {} ../lp_ray_finder_backup/ \;
    
    # Remove temporary files
    rm -f test_cone.*
    rm -f *.tmp
    
    echo "✅ Cleanup completed!"
    echo "📁 Backed up files moved to: ../lp_ray_finder_backup/"
    
    echo ""
    echo "📊 Remaining files:"
    ls -la | grep -v "^d" | wc -l
    echo ""
    
else
    echo "❌ Cleanup cancelled"
fi