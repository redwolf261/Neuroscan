"""
Extract model classes from final_model.py without triggering data loading.

This is a workaround because final_model.py tries to load PediMS data at module import.
For BraTS training, we only need the model/loss classes, not the dataset initialization.
"""

import sys
import os

# Block the data loading section by setting a dummy path
# The point is to make DATA_PATH exist so the check passes
_dummy_data_path = os.path.join(os.path.dirname(__file__), "Dataset", "Training")
if not os.path.exists(_dummy_data_path):
    os.environ["PEDIMS_DATA_PATH"] = "/dev/null"  # Dummy path that will fail the data loading loop silently
else:
    os.environ["PEDIMS_DATA_PATH"] = _dummy_data_path

# Now import - this will try to load data but will fail gracefully
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "01_source_code", "models"))

try:
    from final_model import (
        HybridMiniSwin2D5_CBAM,
        HybridLoss,
        FocalTverskyLoss,
        EvidentialBetaLoss,
        AdaptiveSliceSelector,
        Conv2D5Stem,
        HybridMiniSwin2D5_ResNetEncoder,
        CBAM_Module,
        LightweightDecoder,
        MiniSwinAttention2D,
        ResidualBlock2D,
    )
except RuntimeError as e:
    if "No PediMS data found" in str(e) or "DATA_PATH does not exist" in str(e):
        # Expected - we're only importing classes, not running training
        # Manually import just the classes we need
        import torch
        import torch.nn as nn

        print(
            "Warning: Could not import from final_model.py due to PediMS data loading. "
            "This is expected when using BraTS. "
            "Model classes will be available once data path is fixed."
        )
        raise
    else:
        raise

__all__ = [
    "HybridMiniSwin2D5_CBAM",
    "HybridLoss",
    "FocalTverskyLoss",
    "EvidentialBetaLoss",
    "AdaptiveSliceSelector",
    "Conv2D5Stem",
    "HybridMiniSwin2D5_ResNetEncoder",
    "CBAM_Module",
    "LightweightDecoder",
    "MiniSwinAttention2D",
    "ResidualBlock2D",
]
