"""
NeuroScan model definitions extracted from final_model.py.

This module re-exports the model classes without executing the training/data loading code
that exists at module level in final_model.py.
"""

import sys
import os
import ast
import importlib.util

# Path to final_model.py
FINAL_MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "01_source_code/models/final_model.py"
)


def _extract_and_load_models():
    """
    Extract model class definitions from final_model.py without executing module-level code.
    """
    # Read the source file
    with open(FINAL_MODEL_PATH, 'r') as f:
        source_code = f.read()

    # Parse the AST to find class definitions we need
    tree = ast.parse(source_code)

    # Build a namespace with necessary imports
    namespace = {
        'nn': __import__('torch.nn', fromlist=['Module']),
        'torch': __import__('torch'),
        'F': __import__('torch.nn.functional'),
        'np': __import__('numpy'),
        'DropPath': None,  # Will be defined in the exec
    }

    # Extract and execute only the class definitions (not the module-level code)
    classes_to_extract = [
        'DropPath',
        'AdaptiveSliceSelector',
        'Conv2D5Stem',
        'MiniSwinAttention2D',
        'ResidualBlock2D',
        'HybridMiniSwin2D5_ResNetEncoder',
        'CBAM_Module',
        'LightweightDecoder',
        'HybridMiniSwin2D5_CBAM',
        'FocalTverskyLoss',
        'HybridLoss',
        'EvidentialBetaLoss',
    ]

    # Extract each class definition
    extracted_code = ""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name in classes_to_extract:
            # Get the source lines for this class
            start_line = node.lineno - 1
            end_line = node.end_lineno
            class_source = "\n".join(source_code.split("\n")[start_line:end_line])
            extracted_code += class_source + "\n\n"

    # Try to execute the extracted code
    try:
        exec(extracted_code, namespace)
    except Exception as e:
        print(f"Warning: Could not extract all classes via AST: {e}")
        print("Falling back to direct import (may trigger data loading)...")
        # Fall back to direct import
        spec = importlib.util.spec_from_file_location("final_model", FINAL_MODEL_PATH)
        final_model = importlib.util.module_from_spec(spec)
        sys.modules['final_model'] = final_model
        spec.loader.exec_module(final_model)
        namespace = vars(final_model)

    return namespace


# Load models at module import time
_models = _extract_and_load_models()

# Export the classes
HybridMiniSwin2D5_CBAM = _models.get('HybridMiniSwin2D5_CBAM')
HybridLoss = _models.get('HybridLoss')
EvidentialBetaLoss = _models.get('EvidentialBetaLoss')
FocalTverskyLoss = _models.get('FocalTverskyLoss')

if HybridMiniSwin2D5_CBAM is None:
    raise ImportError("Could not extract HybridMiniSwin2D5_CBAM from final_model.py")
if HybridLoss is None:
    raise ImportError("Could not extract HybridLoss from final_model.py")

__all__ = [
    'HybridMiniSwin2D5_CBAM',
    'HybridLoss',
    'EvidentialBetaLoss',
    'FocalTverskyLoss',
]
