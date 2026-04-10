"""
Pre-Ablation Checklist
======================
Run this before starting your 14-hour ablation study to verify everything is configured correctly.
"""

import os
import sys

# Add color support for Windows
try:
    from colorama import init, Fore, Style
    init()
    GREEN = Fore.GREEN
    RED = Fore.RED
    YELLOW = Fore.YELLOW
    RESET = Style.RESET_ALL
except:
    GREEN = RED = YELLOW = RESET = ""

def check(condition, message, critical=True):
    """Check a condition and print colored result."""
    if condition:
        print(f"{GREEN}✓{RESET} {message}")
        return True
    else:
        symbol = f"{RED}✗{RESET}" if critical else f"{YELLOW}⚠{RESET}"
        print(f"{symbol} {message}")
        return False

print("="*80)
print("ABLATION STUDY PRE-FLIGHT CHECKLIST")
print("="*80)
print()

all_good = True

# 1. Check files exist
print("[1/6] Checking required files...")
all_good &= check(os.path.exists("final_model.py"), "final_model.py exists")
all_good &= check(os.path.exists("run_ablation_study.py"), "run_ablation_study.py exists")
all_good &= check(os.path.exists("ABLATION_WORKFLOW.md"), "ABLATION_WORKFLOW.md exists")
all_good &= check(os.path.exists("ABLATION_QUICK_START.md"), "ABLATION_QUICK_START.md exists")
print()

# 2. Check Google Drive access
print("[2/6] Checking Google Drive...")
drive_paths = [
    "G:\\My Drive",
    os.path.join(os.path.expanduser("~"), "Google Drive"),
    "C:\\Users\\HP\\Google Drive",
]
drive_found = False
for path in drive_paths:
    if os.path.exists(path):
        print(f"{GREEN}✓{RESET} Google Drive found: {path}")
        drive_found = True
        break
all_good &= check(drive_found, "Google Drive accessible")
print()

# 3. Check dataset
print("[3/6] Checking PediMS dataset...")
dataset_path = "G:\\My Drive\\Dataset\\PediMS\\PediMS"
if os.path.exists(dataset_path):
    patients = [d for d in os.listdir(dataset_path) if d.startswith("P")]
    all_good &= check(len(patients) >= 9, f"Found {len(patients)} patients (expected 9)")
else:
    all_good &= check(False, f"Dataset not found: {dataset_path}")
print()

# 4. Check configuration in final_model.py
print("[4/6] Checking final_model.py configuration...")
with open("final_model.py", "r", encoding="utf-8") as f:
    content = f.read()
    
    # Check epochs
    mae_epochs = None
    seg_epochs = None
    for line in content.split("\n"):
        if "MAE_EPOCHS = " in line and not line.strip().startswith("#"):
            try:
                mae_epochs = int(line.split("=")[1].split("#")[0].strip())
            except:
                pass
        if "SEGMENTATION_EPOCHS = " in line and not line.strip().startswith("#"):
            try:
                seg_epochs = int(line.split("=")[1].split("#")[0].strip())
            except:
                pass
    
    if mae_epochs and seg_epochs:
        check(mae_epochs >= 30, f"MAE_EPOCHS = {mae_epochs} (recommend 50+ for ablation)", critical=False)
        check(seg_epochs >= 20, f"SEGMENTATION_EPOCHS = {seg_epochs} (recommend 30+ for ablation)", critical=False)
    else:
        print(f"{YELLOW}⚠{RESET} Could not parse epoch values - check manually")
    
    # Check ablation config
    if "ABLATION_CONFIGS = {" in content:
        print(f"{GREEN}✓{RESET} ABLATION_CONFIGS defined (7 configurations)")
    else:
        all_good &= check(False, "ABLATION_CONFIGS missing")
print()

# 5. Check dependencies
print("[5/6] Checking Python dependencies...")
try:
    import torch
    print(f"{GREEN}✓{RESET} PyTorch installed (version {torch.__version__})")
    if torch.cuda.is_available():
        print(f"{GREEN}✓{RESET} CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print(f"{YELLOW}⚠{RESET} CUDA not available (will use CPU - very slow!)")
except ImportError:
    all_good &= check(False, "PyTorch not installed")

try:
    import monai
    print(f"{GREEN}✓{RESET} MONAI installed")
except ImportError:
    all_good &= check(False, "MONAI not installed")

try:
    import pandas
    print(f"{GREEN}✓{RESET} Pandas installed")
except ImportError:
    all_good &= check(False, "Pandas not installed")

try:
    import scipy
    print(f"{GREEN}✓{RESET} SciPy installed (for statistics)")
except ImportError:
    check(False, "SciPy not installed (needed for significance tests)", critical=False)
print()

# 6. Check disk space
print("[6/6] Checking disk space...")
try:
    import shutil
    stats = shutil.disk_usage("G:\\")
    free_gb = stats.free / (1024**3)
    check(free_gb >= 10, f"Google Drive has {free_gb:.1f} GB free (need ~5GB for checkpoints)", critical=False)
    
    stats_local = shutil.disk_usage("C:\\")
    free_local = stats_local.free / (1024**3)
    check(free_local >= 5, f"C:\\ drive has {free_local:.1f} GB free", critical=False)
except:
    print(f"{YELLOW}⚠{RESET} Could not check disk space")
print()

# Summary
print("="*80)
if all_good:
    print(f"{GREEN}ALL CHECKS PASSED!{RESET}")
    print()
    print("You're ready to start the ablation study.")
    print()
    print("Next steps:")
    print("1. Set MAE_EPOCHS = 50 and SEGMENTATION_EPOCHS = 30 in final_model.py")
    print("2. Set ABLATION_CONFIG = 'baseline'")
    print("3. Run: python final_model.py")
    print("4. Repeat for each of 7 configs (see ABLATION_QUICK_START.md)")
    print("5. Run: python run_ablation_study.py")
else:
    print(f"{RED}ISSUES DETECTED - PLEASE FIX BEFORE STARTING{RESET}")
    print()
    print("Review the ✗ items above and resolve them.")
    print("Then re-run this checklist: python check_ablation_ready.py")
print("="*80)
