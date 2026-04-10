"""
RUN ALL REMAINING TESTS
========================
Executes both CSRF and Noise Robustness tests in sequence

Total Runtime: ~3-4 hours
Tests: 2 (CSRF Variants + Noise Robustness)

Usage:
    python run_all_tests.py
"""

import os
import sys
import subprocess
from datetime import datetime

print("=" * 80)
print("RUN ALL REMAINING TESTS")
print("=" * 80)
print(f"\nStart Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\n📊 Tests to run:")
print("  1. CSRF Variants Analysis (~2-3 hours)")
print("     - 4 fusion methods")
print("     - 15 epochs each")
print("  ")
print("  2. Noise Robustness Analysis (~1 hour)")
print("     - 5 noise types × 4 levels")
print("     - 20 samples")
print("\n⏰ Total Estimated Time: 3-4 hours")
print("\n" + "=" * 80)

# Run tests sequentially
tests = [
    ("CSRF Variants", "run_csrf_test.py"),
    ("Noise Robustness", "run_noise_test.py"),
]

overall_start = datetime.now()
completed = []
failed = []

for test_name, script in tests:
    print("\n" + "=" * 80)
    print(f"STARTING: {test_name}")
    print("=" * 80)
    
    test_start = datetime.now()
    
    try:
        result = subprocess.run(["python", script], check=True)
        
        test_end = datetime.now()
        duration = (test_end - test_start).total_seconds() / 3600
        
        print(f"\n✅ {test_name} completed in {duration:.1f} hours")
        completed.append((test_name, duration))
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {test_name} FAILED with exit code {e.returncode}")
        failed.append(test_name)
        
        response = input(f"\nContinue to next test? (y/n): ")
        if response.lower() != 'y':
            print("\n⚠️ Testing stopped by user")
            break
    
    except KeyboardInterrupt:
        print(f"\n\n⚠️ {test_name} interrupted by user")
        failed.append(test_name)
        
        response = input(f"\nContinue to next test? (y/n): ")
        if response.lower() != 'y':
            print("\n⚠️ Testing stopped by user")
            break

# Final summary
overall_end = datetime.now()
total_duration = (overall_end - overall_start).total_seconds() / 3600

print("\n" + "=" * 80)
print("ALL TESTS COMPLETE!")
print("=" * 80)
print(f"\nStart: {overall_start.strftime('%H:%M:%S')}")
print(f"End: {overall_end.strftime('%H:%M:%S')}")
print(f"Total Duration: {total_duration:.1f} hours")

if completed:
    print(f"\n✅ Completed ({len(completed)}/{len(tests)}):")
    for name, dur in completed:
        print(f"  - {name}: {dur:.1f} hours")

if failed:
    print(f"\n❌ Failed ({len(failed)}/{len(tests)}):")
    for name in failed:
        print(f"  - {name}")

print("\n📂 Results locations:")
print("  - research/csrf_variants_results_quick/")
print("  - research/noise_robustness_results_quick/")

print("\n" + "=" * 80)
