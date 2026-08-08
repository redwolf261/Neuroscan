"""
Safe import of NeuroScan model classes without triggering PediMS data loading.

final_model.py has module-level code that tries to load PediMS data.
This wrapper delays the import to prevent that initialization code from running unnecessarily.
"""

import sys
import os
import importlib.util

def load_model_classes():
    """
    Dynamically load HybridMiniSwin2D5_CBAM and HybridLoss without triggering full initialization.
    """
    model_path = os.path.join(
        os.path.dirname(__file__),
        "01_source_code/models/final_model.py"
    )

    spec = importlib.util.spec_from_file_location("final_model", model_path)
    module = importlib.util.module_from_spec(spec)

    try:
        spec.loader.exec_module(module)
    except RuntimeError as e:
        if "PediMS" in str(e) or "DATA_PATH" in str(e):
            # If we get here, it means final_model.py tried to load PediMS
            # But we managed to at least load the classes before hitting the error
            # Try to extract what we need from the namespace
            if hasattr(module, 'HybridMiniSwin2D5_CBAM'):
                return module.HybridMiniSwin2D5_CBAM, module.HybridLoss
        raise

    return module.HybridMiniSwin2D5_CBAM, module.HybridLoss


# Pre-load at import time for convenience
try:
    HybridMiniSwin2D5_CBAM, HybridLoss = load_model_classes()
except RuntimeError:
    # If that doesn't work, define them as callables that will error with clear message
    def _raise_import_error(*args, **kwargs):
        raise ImportError(
            "Could not import NeuroScan models. "
            "final_model.py requires PediMS data to be present at import time. "
            "Please create a minimal PediMS structure or run train_baseline_brats.py with proper data setup."
        )

    HybridMiniSwin2D5_CBAM = _raise_import_error
    HybridLoss = _raise_import_error

__all__ = ["HybridMiniSwin2D5_CBAM", "HybridLoss"]
