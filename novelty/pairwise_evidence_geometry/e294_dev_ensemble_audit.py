"""E294-Dev: Multi-Model Ensemble Audit & Threshold Tuning on Development Data.

Strict Development Cohort Evaluation (13 subjects):
Zero leakage to locked test set.

Evaluates:
- M1: E131_v5control_seed0 (Frozen Production Baseline)
- M2: E141_evidence_seed0
- M3: E131_v14_seed0
- M4: E141_shuffled_seed0
- Ensembles:
    - Ens_2 (M1 + M2)
    - Ens_3 (M1 + M2 + M3)
    - Ens_4 (M1 + M2 + M3 + M4)
Across:
- Raw soft average vs Hierarchical containment (ET <= WT)
- Threshold sweeps for ET Dice.
"""

import sys, os, time, csv, json
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT_REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT_REPO))
sys.path.insert(0, str(ROOT_REPO / 'experiments' / 'exp_e12_eggo_m' / 'e131'))

from neuroscan_3d_v5 import UNet3D_v5
from neuroscan_3d_v13 import UNet3D_v14
from e257_common import load_populations, build_splits, ROOT, PATCH
from Dataset.brats_multimodal_dataset import BraTSMultimodalDataset
from e245_dec1_internal_decomposition import load_patch

ET, TC, WT = 0, 1, 2

def dice_score(pred, true):
    tp = (pred & true).sum()
    fp = (pred & ~true).sum()
    fn = (~pred & true).sum()
    if true.sum() == 0 and pred.sum() == 0:
        return 1.0
    return float(2 * tp / max(2 * tp + fp + fn, 1e-8))

def main():
    dev = torch.device('cuda')
    print('=' * 80)
    print('E294-DEV: MULTI-MODEL ENSEMBLE AUDIT ON DEVELOPMENT COHORT (13 SUBJECTS)')
    print('=' * 80)

    # 1. Load models
    models = {}
    print('Loading M1: E131_v5control_seed0...')
    m1 = UNet3D_v5(4, 3).to(dev).eval()
    ck1 = torch.load(ROOT / 'experiments/exp_e12_eggo_m/e131/runs/E131_v5control_seed0/checkpoints/best.pth', map_location=dev, weights_only=False)
    m1.load_state_dict(ck1['model_state'])
    models['M1_v5control'] = m1

    print('Loading M2: E141_evidence_seed0...')
    m2 = UNet3D_v5(4, 3).to(dev).eval()
    ck2 = torch.load(ROOT / 'experiments/exp_e12_eggo_m/e141/runs/E141_evidence_seed0/checkpoints/best.pth', map_location=dev, weights_only=False)
    m2.load_state_dict(ck2['model_state'])
    models['M2_evidence'] = m2

    print('Loading M3: E131_v14_seed0...')
    m3 = UNet3D_v14(4, 3).to(dev).eval()
    ck3 = torch.load(ROOT / 'experiments/exp_e12_eggo_m/e131/runs/E131_v14_seed0/checkpoints/best.pth', map_location=dev, weights_only=False)
    m3.load_state_dict(ck3['model_state'])
    models['M3_v14'] = m3

    print('Loading M4: E141_shuffled_seed0...')
    m4 = UNet3D_v5(4, 3).to(dev).eval()
    ck4 = torch.load(ROOT / 'experiments/exp_e12_eggo_m/e141/runs/E141_shuffled_seed0/checkpoints/best.pth', map_location=dev, weights_only=False)
    m4.load_state_dict(ck4['model_state'])
    models['M4_shuffled'] = m4

    # 2. Get Dev split
    det_lesions, g2a_lesions, _ = load_populations()
    splits = build_splits(det_lesions, g2a_lesions)
    all_dev_sids = sorted(splits['det_test'])
    rng = np.random.default_rng(290)
    shuffled_dev = rng.permutation(all_dev_sids)
    n_dev = len(shuffled_dev)
    n_train = int(n_dev * 0.6)
    n_val = int(n_dev * 0.2)
    dev_val_sids = list(shuffled_dev[n_train:n_train + n_val])

    ds = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'train', val_split=0.0, patch_size=PATCH)
    sid_to_idx = {os.path.basename(sd): i for i, sd in enumerate(ds.subject_dirs)}
    valid_sids = [sid for sid in dev_val_sids if sid in sid_to_idx]
    print(f'Active Dev Validation Subjects: {len(valid_sids)}')

    # 3. Predict on all 13 subjects
    # Store predictions for each model: dict of sid -> dict of 'probs' (3, D, H, W)
    sub_preds = {m_name: {} for m_name in models}
    sub_targets = {}

    t0 = time.time()
    for sid in valid_sids:
        img_c, tgt_c = load_patch(ds, sid_to_idx, sid)
        img_t = torch.from_numpy(img_c).unsqueeze(0).float().to(dev)
        sub_targets[sid] = (tgt_c > 0.5)

        with torch.no_grad():
            for m_name, model in models.items():
                out = model(img_t)
                p = out['probs'][0].cpu().numpy() # (3, D, H, W)
                sub_preds[m_name][sid] = p
        del img_t
        torch.cuda.empty_cache()

    print(f'Computed forward passes in {time.time() - t0:.1f}s.')

    # 4. Evaluate Individual Models (tau = 0.50)
    print('\n' + '=' * 60)
    print('INDIVIDUAL MODELS PERFORMANCE ON DEV (tau=0.50):')
    print('=' * 60)
    for m_name in models:
        et_dices, tc_dices, wt_dices = [], [], []
        for sid in valid_sids:
            p = sub_preds[m_name][sid]
            t = sub_targets[sid]
            et_dices.append(dice_score(p[ET] >= 0.5, t[ET]))
            tc_dices.append(dice_score(p[TC] >= 0.5, t[TC]))
            wt_dices.append(dice_score(p[WT] >= 0.5, t[WT]))
        print(f'{m_name:18s} | ET: {np.mean(et_dices):.4f} | TC: {np.mean(tc_dices):.4f} | WT: {np.mean(wt_dices):.4f} | Mean: {np.mean([np.mean(et_dices), np.mean(tc_dices), np.mean(wt_dices)]):.4f}')

    # 5. Build Ensembles
    ensembles = {
        'Ens_2 (M1+M2)': ['M1_v5control', 'M2_evidence'],
        'Ens_2 (M1+M3)': ['M1_v5control', 'M3_v14'],
        'Ens_3 (M1+M2+M3)': ['M1_v5control', 'M2_evidence', 'M3_v14'],
        'Ens_4 (M1+M2+M3+M4)': ['M1_v5control', 'M2_evidence', 'M3_v14', 'M4_shuffled'],
    }

    base_et_mean = np.mean([dice_score(sub_preds['M1_v5control'][sid][ET] >= 0.5, sub_targets[sid][ET]) for sid in valid_sids])

    print('\n' + '=' * 80)
    print(f'ENSEMBLE AUDIT (tau=0.50 vs Baseline M1 ET={base_et_mean:.4f}):')
    print('=' * 80)
    ens_probs = {}
    for ens_name, m_list in ensembles.items():
        ens_probs[ens_name] = {}
        et_dices, tc_dices, wt_dices = [], [], []
        for sid in valid_sids:
            p_stack = np.stack([sub_preds[m][sid] for m in m_list], axis=0) # (M, 3, D, H, W)
            p_mean = np.mean(p_stack, axis=0)
            ens_probs[ens_name][sid] = p_mean

            t = sub_targets[sid]
            et_dices.append(dice_score(p_mean[ET] >= 0.5, t[ET]))
            tc_dices.append(dice_score(p_mean[TC] >= 0.5, t[TC]))
            wt_dices.append(dice_score(p_mean[WT] >= 0.5, t[WT]))

        m_et = np.mean(et_dices)
        delta_et = m_et - base_et_mean
        print(f'{ens_name:22s} | ET: {m_et:.4f} ({delta_et:+.4f}) | TC: {np.mean(tc_dices):.4f} | WT: {np.mean(wt_dices):.4f}')

    # 6. Ensemble + Hierarchical Sweeps
    print('\n' + '=' * 85)
    print('OPTIMIZATION: ENSEMBLE + HIERARCHICAL MASKING (ET <= WT) + TAU SWEEP')
    print('=' * 85)
    best_config = None
    best_gain = -999.0

    tau_list = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

    for ens_name in ensembles:
        print(f'\n--- {ens_name} ---')
        for tau_et in tau_list:
            for tau_wt in [0.40, 0.50]:
                dices = []
                for sid in valid_sids:
                    p = ens_probs[ens_name][sid]
                    t = sub_targets[sid]
                    pred = (p[ET] >= tau_et) & (p[WT] >= tau_wt)
                    dices.append(dice_score(pred, t[ET]))
                m_dice = np.mean(dices)
                gain = m_dice - base_et_mean
                if gain > best_gain:
                    best_gain = gain
                    best_config = (ens_name, tau_et, tau_wt, m_dice, gain)
                if tau_et in [0.20, 0.30, 0.40, 0.50]:
                    print(f'  tau_et={tau_et:.2f}, tau_wt={tau_wt:.2f} -> ET Mean Dice: {m_dice:.4f} (Delta: {gain:+.4f}, {gain*100:+.2f}%)')

    print('\n' + '=' * 80)
    print('BEST DEVELOPMENT CONFIGURATION:')
    print(f'Ensemble: {best_config[0]}')
    print(f'Parameters: tau_et={best_config[1]:.2f}, tau_wt={best_config[2]:.2f}')
    print(f'Mean ET Dice: {best_config[3]:.4f} (Baseline: {base_et_mean:.4f})')
    print(f'Net ET Dice Gain on Dev: {best_config[4]:+.4f} ({best_config[4]*100:+.2f}%)')
    print('=' * 80)

if __name__ == '__main__':
    main()
