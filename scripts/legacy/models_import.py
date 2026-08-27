"""
Safe model imports without triggering PediMS data loading.

final_model.py tries to load PediMS data on import, which breaks when training on BraTS.
This module provides a clean way to import just the model classes.
"""

import sys
import os

# Temporarily disable data loading by setting a flag
os.environ["SKIP_PEDIMS_LOADING"] = "1"

# Add path for final_model imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "01_source_code", "models"))

# This will fail because final_model tries to load PediMS
# We need to patch final_model.py or create minimal model definitions

# For now, we'll define minimal versions of the classes needed
# based on what final_model.py exports
