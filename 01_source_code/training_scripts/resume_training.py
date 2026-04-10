#!/usr/bin/env python
"""
Minimal launcher to resume training with delayed imports
"""
import os
import sys
import time

# Set environment variables before any imports
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

print("=" * 80)
print("RESUMING TRAINING - Loading dependencies...")
print("=" * 80)

# Small delay to let filesystem settle
time.sleep(1)

# Now import and run the main script
if __name__ == "__main__":
    import final_model
