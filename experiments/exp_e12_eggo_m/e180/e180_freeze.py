"""
E180 Stage 0 -- Freeze manifest.

Pre-registered in docs/phases/PHASE_E180_FREEZE_MANIFEST.md. Read that first.

No scientific computation. Records checkpoint identity, the 125-subject validation
ID list, code commit, and environment info into
experiments/exp_e12_eggo_m/e180/FROZEN/MANIFEST.json, so every later E180 stage can
verify it is running against the exact same frozen state.
"""
import sys
import json
import hashlib
import platform
import subprocess
from pathlib import Path

import torch

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from Dataset.brats_multimodal_dataset import create_multimodal_loaders  # noqa: E402

OUT_DIR = Path(__file__).parent
FROZEN_DIR = OUT_DIR / "FROZEN"

CKPT = (project_root / "experiments" / "exp_e12_eggo_m" / "e131" / "runs"
        / "E131_v5control_seed0" / "checkpoints" / "best.pth")
EXPECTED_DICE = 0.8929357248544694
PATCH = (128, 128, 128)
OVERLAP = 0.25
TARGET_STAGE = "enc3"


def sha256_16(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def sha256_16_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(project_root))
        return out.decode().strip()
    except Exception as e:
        return f"UNKNOWN ({e})"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- checkpoint identity ----
    ckpt = torch.load(str(CKPT), map_location=device, weights_only=False)
    got = ckpt.get("best_mean_dice")
    assert got is not None and abs(float(got) - EXPECTED_DICE) < 1e-9, \
        f"Checkpoint identity FAILED: expected {EXPECTED_DICE}, got {got}"
    print(f"[Sanity] checkpoint identity PASS (best_mean_dice={got}).")
    ckpt_hash = sha256_16_file(CKPT)

    # ---- dataset identity ----
    _, val_loader = create_multimodal_loaders(
        root_dir=str(project_root / "Dataset" / "Training"),
        batch_size=1, num_workers=0, val_split=0.1, patch_size=PATCH, seed=0)
    n_val = len(val_loader.dataset)
    assert n_val == 125, f"Expected 125 validation subjects, got {n_val}"

    sids = []
    for i in range(n_val):
        _, _, sid = val_loader.dataset[i]
        sids.append(sid)
    sids_sorted = sorted(sids)
    sid_hash = sha256_16("\n".join(sids_sorted).encode())
    print(f"[Sanity] {n_val} validation subjects, ID list hash={sid_hash}")

    # ---- environment ----
    manifest = {
        "checkpoint_path": str(CKPT),
        "checkpoint_sha256_16": ckpt_hash,
        "checkpoint_best_mean_dice": float(got),
        "dataset_root": str(project_root / "Dataset" / "Training"),
        "dataset_val_split": 0.1,
        "dataset_seed": 0,
        "n_val_subjects": n_val,
        "subject_id_list_sha256_16": sid_hash,
        "subject_ids": sids_sorted,
        "code_commit": git_commit(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "sliding_window_overlap": OVERLAP,
        "patch_size": list(PATCH),
        "target_stage": TARGET_STAGE,
        "note": "FROZEN. E180 Stage 0. No modification to E126-E176 artifacts.",
    }

    FROZEN_DIR.mkdir(parents=True, exist_ok=True)
    with open(FROZEN_DIR / "MANIFEST.json", "w") as f:
        json.dump(manifest, f, indent=1)

    print("\n" + "=" * 70)
    print(json.dumps({k: v for k, v in manifest.items() if k != "subject_ids"}, indent=1))
    print("=" * 70)
    print(f"\nWrote {FROZEN_DIR / 'MANIFEST.json'}")


if __name__ == "__main__":
    main()
