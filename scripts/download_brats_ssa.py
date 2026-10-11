"""
Download BraTS 2023 Sub-Saharan Africa (BraTS-Africa / SSA) Dataset from Hugging Face
into Dataset/BraTS2023_SSA/ for external cross-dataset evaluation.
"""

import os
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent.parent
TARGET_DIR = ROOT / 'Dataset' / 'BraTS2023_SSA'
TARGET_DIR.mkdir(parents=True, exist_ok=True)

print("Starting download of BraTS2023-SSA dataset from Hugging Face...")
repo_dir = snapshot_download(
    repo_id="Angelou0516/brats2023-ssa",
    repo_type="dataset",
    local_dir=str(TARGET_DIR / "_hf_raw"),
    local_dir_use_symlinks=False,
    max_workers=4
)

# Unpack/move folders into clean layout: Dataset/BraTS2023_SSA/<SubjectID>/<files>
raw_sub_dir = TARGET_DIR / "_hf_raw" / "ASNR-MICCAI-BraTS2023-SSA-Challenge-TrainingData_V2"
if raw_sub_dir.exists():
    for sub in raw_sub_dir.iterdir():
        if sub.is_dir():
            dest = TARGET_DIR / sub.name
            if not dest.exists():
                shutil.move(str(sub), str(dest))
    shutil.rmtree(TARGET_DIR / "_hf_raw", ignore_errors=True)

subjects = [d for d in TARGET_DIR.iterdir() if d.is_dir() and d.name.startswith("BraTS-SSA")]
print(f"Successfully prepared {len(subjects)} BraTS-SSA external subjects at: {TARGET_DIR}")
