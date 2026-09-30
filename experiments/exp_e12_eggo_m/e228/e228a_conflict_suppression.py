"""E228-A -- causal intervention test of E227's directional-conflict
mechanism. NOT a Dice-improvement campaign; the primary endpoint is the
TRAJECTORY of a fixed, pre-registered set of lesions, per the user's own
explicit design (see novelty/pairwise_evidence_geometry/E228_lesion_sets.csv,
built from E227_gradient.csv BEFORE this script runs, frozen, never
re-selected using any outcome from this run).

THREE CONDITIONS, matching the user's exact spec (a fourth "random
suppression" control is not optional -- without it, any Dice/probability
gain from suppression is uninterpretable, since it could mean "any D4
reduction helps" rather than "resolving THIS conflict helps"):

  1. BASELINE       -- unmodified v5 production loss (seg + boundary + D4),
                       exactly as train_e130_multimodal_baseline.py computes it.
  2. CONFLICT        -- for subjects containing a frozen CONFLICT-POSITIVE
                       lesion (E227's high-D4-gradient, negative-cosine set,
                       n=18), that lesion's own D4 cells are EXCLUDED from
                       the D4 loss (contribute zero) for that training step.
                       All other patches/lesions/subjects: baseline D4 loss,
                       UNCHANGED.
  3. RANDOM          -- IDENTICAL mechanism, but applied to the frozen
                       RANDOM-CONTROL lesion set (n=18, size-matched to
                       conflict-positive, NOT selected on cosine) instead.

WHY NO LIVE GRADIENT-COSINE COMPUTATION: conflict/random status is FROZEN
from E227's own diagnostic measurement (single dual-backward-pass audit on
the E131 checkpoint). This experiment does NOT recompute cosine during
training -- it tests whether suppressing D4 on lesions ALREADY IDENTIFIED
as conflicting (vs. a matched set NOT so identified) changes their
trajectory. This is the "cheapest faithful approximation" the user asked
for, explicitly NOT a candidate production mechanism (a real fix would need
live/recurring conflict detection, which is out of scope here and would
require its own novelty audit before ever being proposed).

TRACKED PER EPOCH, for EVERY lesion in EITHER frozen set (36 total) plus a
comparison "all other missed" pool:
  - mean predicted ET probability over the lesion's own voxels (PRIMARY,
    per the user's explicit rationale: probability is continuous/gradual,
    detection is binary and can jump abruptly)
  - lesion-level Dice
  - binary detection at tau=0.5
  - max ET probability over the lesion

ALSO TRACKED (secondary/project endpoints): whole-val-set ET/TC/WT Dice,
ET FP volume, per the user's explicit "primary mechanistic endpoint vs
primary project endpoint" distinction.

Short run (5 epochs, matching this project's own E224-R/E103 short-audit
convention) -- this is NOT a Dice-optimization campaign.
"""
import sys, os, csv, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from neuroscan_3d_v5 import UNet3D_v5  # noqa: E402
from Dataset.brats_multimodal_dataset import BraTSMultimodalDataset, create_multimodal_loaders  # noqa: E402

HERE = Path(__file__).parent
LESION_SETS_CSV = ROOT / 'novelty/pairwise_evidence_geometry/E228_lesion_sets.csv'
CKPT_INIT = ROOT / 'experiments/exp_e12_eggo_m/e131/runs/E131_v5control_seed0/checkpoints/best.pth'

SEED = 0
N_EPOCHS = 5
PATCH = (128, 128, 128)
BATCH_SIZE = 1
# NUM_WORKERS restored to a real value: OffsetTrackingDataset returns each
# patch's native-volume crop start offset as part of its own __getitem__
# tuple (see class def below), which survives normal DataLoader pickling/
# collation across worker processes -- unlike the earlier num_workers=0-
# only approach (an instance-attribute side-channel, invisible once
# num_workers>0 spawns separate processes; that version measured 935s/
# epoch vs E224-R's ~200s at num_workers=4, confirmed I/O-bound via GPU
# utilization stuck at 46%).
NUM_WORKERS = 4
LR = 1e-4
WD = 1e-5
ET = 0
K = 4
MU = 0.1              # boundary weight, matches production
LAMBDA_DS3 = 0.9927    # D4 weight, matches production
SMOKE_SUBJECTS = None
# AUDIT_SUBJECTS: full-1126-subject run measured 751s/epoch x 5 epochs x
# 3 conditions ~= 3.1h -- too slow for this trajectory-diagnostic's actual
# purpose (per user decision). Capped to 300, matching E224-R's own
# established short-audit convention -- but see load_capped_subject_dirs
# below: a plain first-300 slice would NOT guarantee the 31 tracked
# subjects (whose lesions this whole experiment exists to measure) are
# actually included, since they were not selected by dataset order.
AUDIT_SUBJECTS = 300


def set_seed(s):
    import random
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def focal_tversky(probs, target, alpha=0.5, beta=0.5, gamma=4.0 / 3.0, eps=1e-6):
    """UNCHANGED copy of train_e130's own function -- verbatim, not
    reimplemented, to guarantee the baseline condition is bit-identical to
    production."""
    dims = tuple(range(2, probs.dim()))
    tp = (probs * target).sum(dims)
    fp = (probs * (1 - target)).sum(dims)
    fn = ((1 - probs) * target).sum(dims)
    ti = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return ((1 - ti) ** gamma).mean()


def evidential_beta(alpha, beta, target, weight=0.5, eps=1e-6):
    """UNCHANGED copy of train_e130's own function."""
    s = alpha + beta
    p = alpha / (s + eps)
    nll = (target - p) ** 2 + p * (1 - p) / (s + 1.0)
    return weight * nll.mean()


class _RngRecorder:
    """Wraps dataset.rng (a numpy Generator) so every .random()/.integers()
    call is passed through UNCHANGED to the real rng (identical values,
    identical state advancement -- production behavior is byte-for-byte
    preserved) while also being logged in order. This lets us recompute
    _sample_patch's exact centre/offset OUTSIDE the original function,
    from the SAME recorded draws, without reimplementing or duplicating
    its selection logic -- avoiding the risk of the reimplementation
    silently drifting from production if _sample_patch ever changes."""
    def __init__(self, real_rng):
        self._real = real_rng
        self.log = []

    def random(self, *a, **kw):
        v = self._real.random(*a, **kw)
        self.log.append(('random', v))
        return v

    def integers(self, *a, **kw):
        v = self._real.integers(*a, **kw)
        self.log.append(('integers', v))
        return v

    def __getattr__(self, name):
        # GUARD against infinite recursion during unpickling in a spawned
        # worker process: Python's pickle protocol probes for dunder
        # attributes (__reduce__, __getstate__, __setstate__, etc.) via
        # getattr BEFORE __init__ has run and set self._real. Without this
        # guard, __getattr__('_real') recurses into itself forever trying
        # to find '_real' via getattr(self._real, '_real') on an object
        # that doesn't have '_real' set yet -- caught via smoke test
        # (RecursionError, worker process crash) before this was fixed.
        if name == '_real':
            raise AttributeError(name)
        return getattr(self._real, name)


class OffsetTrackingDataset(BraTSMultimodalDataset):
    """Subclass that ALSO returns each patch's native-volume crop start
    offset as a 4th item, so it survives a num_workers>0 DataLoader's
    process boundary via normal pickling/collation -- unlike an instance
    attribute (the earlier num_workers=0-only approach this replaces),
    which lives only in whichever worker process set it and is invisible
    to the main process once num_workers>0 spawns separate processes.

    Uses the SAME _RngRecorder + recompute_offset_from_log machinery as
    before (verified correct: 0 mismatches across 20 direct checks
    against a manual native-volume crop, see this session's own testing)
    -- only WHERE it is installed changes (per-instance in __init__,
    which runs once per worker process at DataLoader startup, not a
    live patch of a shared instance across a process boundary)."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.rng = _RngRecorder(self.rng)

    def __getitem__(self, idx):
        if self.split == 'val':
            return super().__getitem__(idx) + ((-1, -1, -1),)
        image, target, sid = self._load_subject(self.subject_dirs[idx])
        self.rng.log.clear()
        img, tgt = self._sample_patch(image, target)
        offset = recompute_offset_from_log(
            self.rng.log, image, target, self.patch_size, self.fg_bias)
        return torch.from_numpy(img).float(), torch.from_numpy(tgt).float(), sid, offset


def recompute_offset_from_log(log, image, target, patch_size, fg_bias):
    """Recomputes (z,y,x) from the RECORDED rng draws, using arithmetic
    copied VERBATIM from Dataset/brats_multimodal_dataset.py's own
    _sample_patch (verified against the real source immediately before
    writing this, not from memory) -- NOT a reimplementation of the
    SELECTION policy (want_fg threshold, fg-vs-brain fallback), just a
    faithful replay of the same arithmetic given the same recorded random
    values, so this cannot silently drift: if _sample_patch's arithmetic
    ever changes, this would need updating too, but the draws themselves
    are guaranteed identical since we recorded the REAL rng's own outputs."""
    D, H, W = image.shape[1:]
    pd, ph, pw = patch_size
    pd, ph, pw = min(pd, D), min(ph, H), min(pw, W)
    idx = 0
    want_fg = log[idx][1] < fg_bias; idx += 1
    centre = None
    if want_fg:
        fg = np.argwhere(target[2] > 0)
        if len(fg):
            i = log[idx][1]; idx += 1
            centre = fg[i]
    if centre is None:
        brain = np.argwhere(image[0] != 0)
        if len(brain):
            i = log[idx][1]; idx += 1
            centre = brain[i]
        else:
            centre = np.array([D // 2, H // 2, W // 2])
    starts = []
    for c, p, full in zip(centre, (pd, ph, pw), (D, H, W)):
        s = int(c) - p // 2
        starts.append(int(np.clip(s, 0, max(0, full - p))))
    return tuple(starts)


def capped_subject_dirs(all_dirs, cap, tracked_subjects):
    """Returns the first `cap` subjects from `all_dirs` (same deterministic
    order E224-R itself relied on -- fixed RandomState(42) split, identical
    every construction), but with any TRACKED subject not naturally in that
    first-`cap` slice swapped in, replacing the lowest-priority non-tracked
    subject at the tail. Guarantees every one of E227's 31 tracked subjects
    (the whole reason this experiment exists) is actually trained on,
    while keeping the cap size and the rest of the subject list identical
    to a plain first-`cap` slice wherever possible -- i.e. this is a
    minimal, disclosed deviation from "just take the first 300", not a
    silent one."""
    base = list(all_dirs[:cap])
    base_ids = set(os.path.basename(d) for d in base)
    missing = [d for d in all_dirs if os.path.basename(d) in tracked_subjects
              and os.path.basename(d) not in base_ids]
    if missing:
        print(f'  capped_subject_dirs: {len(missing)} tracked subjects not in '
              f'first {cap}, force-including (replacing tail entries)', flush=True)
        base = base[:len(base) - len(missing)] + missing
    return base


def load_lesion_sets():
    """Returns {subject_id: [{'comp_id':.., 'group':.., 'size':..}, ...]}
    -- FROZEN, read-only, built by a separate script BEFORE this run from
    E227's own diagnostic measurements. This function does not compute or
    select anything -- it only loads a pre-existing file."""
    by_subject = {}
    with open(LESION_SETS_CSV) as f:
        for row in csv.DictReader(f):
            by_subject.setdefault(row['subject_id'], []).append({
                'comp_id': int(row['comp_id']), 'group': row['group'],
                'size': float(row['size'])})
    return by_subject


def d4_cell_mask_for_lesion(target_et_native, comp_id, start, patch_size, K, dev):
    """Recomputes the SAME connected-component labeling E227/E226/E223 all
    use (ndi.label(Y[ET]), MIN_VOX=5 filter implicit since these lesions
    already passed that filter when the frozen set was built) to find
    THIS lesion's voxels, crops to the given patch start, and returns a
    (1,1,grid,grid,grid) boolean D4-cell mask -- True for cells this
    lesion's own voxels touch, in PATCH-LOCAL D4 coordinates. Returns None
    if this lesion's voxels are not actually present in this crop (should
    not happen when called correctly, i.e. only for the exact subject/
    patch this lesion was tracked for building the loaders, but guarded)."""
    from scipy import ndimage
    Y_et = target_et_native > 0.5
    et_lbl, et_n = ndimage.label(Y_et)
    if comp_id < 1 or comp_id > et_n:
        return None
    cm = et_lbl == comp_id
    lz, ly, lx = np.where(cm)
    if len(lz) == 0:
        return None
    z0, y0, x0 = start
    pd, ph, pw = patch_size
    local_z = lz - z0; local_y = ly - y0; local_x = lx - x0
    inside = ((local_z >= 0) & (local_z < pd) &
             (local_y >= 0) & (local_y < ph) & (local_x >= 0) & (local_x < pw))
    if not inside.any():
        return None
    grid = pd // K
    cz = local_z[inside] // K; cy = local_y[inside] // K; cx = local_x[inside] // K
    mask = np.zeros((grid, grid, grid), dtype=np.float32)
    mask[cz, cy, cx] = 1.0
    return torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(dev)


def d4_loss_with_suppression(aux_probs3, t_d4, suppress_masks):
    """Computes the SAME per-region-averaged D4 FocalTversky as production
    (focal_tversky(aux_probs3, t_d4), 3-region averaged), EXCEPT for the ET
    channel, where any cells in `suppress_masks` (a list of (1,1,g,g,g)
    boolean-ish tensors, one per tracked lesion present in this patch) are
    ZEROED in both aux_probs3[ET] and t_d4[ET] before the loss is computed
    -- exactly the same effect as E227's masked-loss technique, but
    INVERTED (excluding cells instead of including only them), and applied
    to ALL 3 region channels' shared computation only via the ET channel's
    own modified copy (TC/WT channels are always unmodified, since the
    frozen lesion sets are ET-specific per E226/E227's own scope)."""
    if not suppress_masks:
        return focal_tversky(aux_probs3, t_d4)

    combined_suppress = suppress_masks[0]
    for m in suppress_masks[1:]:
        combined_suppress = torch.clamp(combined_suppress + m, max=1.0)
    keep_mask = 1.0 - combined_suppress  # (1,1,g,g,g), 0 where suppressed

    aux_et = aux_probs3[:, ET:ET + 1] * keep_mask
    t_d4_et = t_d4[:, ET:ET + 1] * keep_mask
    aux_mod = aux_probs3.clone()
    t_d4_mod = t_d4.clone()
    aux_mod[:, ET:ET + 1] = aux_et
    t_d4_mod[:, ET:ET + 1] = t_d4_et
    return focal_tversky(aux_mod, t_d4_mod)


def run_condition(mode, run_name, dev, lesion_sets):
    """mode: 'baseline' | 'conflict' | 'random' -- which frozen lesion set
    (if any) gets its D4 contribution suppressed."""
    assert mode in ('baseline', 'conflict', 'random')
    set_seed(SEED)
    exp_dir = HERE / run_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    model = UNet3D_v5(in_channels=4, out_channels=3).to(dev)
    ck = torch.load(str(CKPT_INIT), map_location=dev, weights_only=False)
    model.load_state_dict(ck['model_state'])
    # NOTE: starts from the FROZEN E131 checkpoint (not a fresh random
    # init) -- this is deliberate: E227's conflict/random lesion sets were
    # diagnosed ON this exact checkpoint's behavior, so the intervention
    # must be tested continuing FROM that same state, not confounded by a
    # completely different training trajectory from scratch.
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=1e-6)
    boundary_criterion = nn.BCEWithLogitsLoss()

    # Built manually with OffsetTrackingDataset (not create_multimodal_
    # loaders) so num_workers>0 works correctly -- the offset now travels
    # back through the DataLoader's normal pickling/collation as part of
    # each item's own return tuple, not via a live instance-attribute
    # mutation (which silently breaks across process boundaries).
    from torch.utils.data import DataLoader
    train_ds = OffsetTrackingDataset(str(ROOT / 'Dataset' / 'Training'), 'train',
                                     val_split=0.1, patch_size=PATCH, fg_bias=0.66, seed=SEED)
    val_ds_raw = OffsetTrackingDataset(str(ROOT / 'Dataset' / 'Training'), 'val',
                                       val_split=0.1, patch_size=PATCH, fg_bias=0.66, seed=SEED)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds_raw, batch_size=1, shuffle=False,
                            num_workers=max(0, NUM_WORKERS // 2), pin_memory=True)

    cap = SMOKE_SUBJECTS if SMOKE_SUBJECTS is not None else AUDIT_SUBJECTS
    if cap is not None:
        tracked_subjects_for_cap = set(lesion_sets.keys())
        train_loader.dataset.subject_dirs = capped_subject_dirs(
            train_loader.dataset.subject_dirs, cap, tracked_subjects_for_cap)

    # PERFORMANCE: precompute a subject_id -> dataset index lookup ONCE,
    # instead of a linear os.path.basename+.index() scan over all subjects
    # on every training step where a tracked subject is hit (~1126-subject
    # scan x up to ~300 steps/epoch x 5 epochs, avoidable entirely).
    sid_to_idx = {os.path.basename(sd): i for i, sd in enumerate(train_loader.dataset.subject_dirs)}

    target_group = {'conflict': 'conflict_positive', 'random': 'random_control'}.get(mode)
    tracked_subjects = set(lesion_sets.keys())

    print(f'[{run_name}] mode={mode}  train={len(train_loader.dataset)} '
          f'val={len(val_loader.dataset)}  tracked_subjects={len(tracked_subjects)}', flush=True)

    metrics_f = open(exp_dir / 'epoch_metrics.csv', 'w', newline='')
    metrics_w = csv.writer(metrics_f)
    metrics_w.writerow(['epoch', 'train_loss', 'val_dice_ET', 'val_dice_TC', 'val_dice_WT',
                        'val_fp_ET', 'epoch_time_sec'])

    traj_f = open(exp_dir / 'lesion_trajectory.csv', 'w', newline='')
    traj_w = csv.writer(traj_f)
    traj_w.writerow(['epoch', 'subject_id', 'comp_id', 'group', 'size',
                     'mean_prob', 'max_prob', 'lesion_dice', 'detected'])

    pd_, ph_, pw_ = PATCH

    for epoch in range(N_EPOCHS):
        t0 = time.time()
        model.train()
        tot, n = 0.0, 0
        pbar = tqdm(train_loader, desc=f'[{run_name}] ep{epoch+1} train')
        for images, targets, sids, offsets in pbar:
            images = images.to(dev, non_blocking=True)
            targets = targets.to(dev, non_blocking=True)
            sid = sids[0] if isinstance(sids, (list, tuple)) else sids
            # offsets: DataLoader's default collation of a 3-tuple field
            # produces [tensor([z,...]), tensor([y,...]), tensor([x,...])]
            # -- i.e. one tensor PER COORDINATE POSITION (each of length
            # batch_size), NOT one tuple per batch item. Verified directly
            # (a toy DataLoader test) before trusting this, since the
            # intuitive-but-wrong assumption (offsets[0] = first sample's
            # tuple) silently produced a 1-element-unpack crash caught by
            # this session's own verification script before the real run.
            start = tuple(int(offsets[i][0]) for i in range(3))

            optimizer.zero_grad(set_to_none=True)
            out = model(images)
            probs, alpha, beta = out['probs'], out['alpha'], out['beta']
            seg = 0.5 * focal_tversky(probs, targets) + 0.5 * evidential_beta(alpha, beta, targets)
            bnd = boundary_criterion(out['boundary_logit'], targets)
            t_d4 = F.avg_pool3d(targets, kernel_size=K, stride=K)

            suppress_masks = []
            if mode != 'baseline' and sid in tracked_subjects:
                # NOTE: _load_subject caches per-subject (cache_subjects
                # default) -- native target needed for correct comp_id
                # labeling. Re-load via the dataset's own cached path.
                _, target_native, _ = train_loader.dataset._load_subject(
                    train_loader.dataset.subject_dirs[sid_to_idx[sid]])
                for lesion in lesion_sets.get(sid, []):
                    if lesion['group'] != target_group:
                        continue
                    cm_mask = d4_cell_mask_for_lesion(
                        target_native[ET], lesion['comp_id'], start, PATCH, K, dev)
                    if cm_mask is not None:
                        suppress_masks.append(cm_mask)

            aux3 = d4_loss_with_suppression(out['aux_probs3'], t_d4, suppress_masks)
            loss = seg + MU * bnd + LAMBDA_DS3 * aux3

            if not torch.isfinite(loss):
                raise RuntimeError(f'non-finite loss at epoch {epoch}, subject {sid}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tot += loss.item(); n += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        scheduler.step()
        train_loss = tot / max(1, n)

        # ---- validation: whole-val-set proxy Dice (patch-level, matching
        # E224-R's own short-audit convention) ----
        model.eval()
        dices = {r: [] for r in range(3)}
        fps = {r: 0 for r in range(3)}
        val_ds = val_loader.dataset
        n_val_patches = min(20, len(val_ds))
        with torch.no_grad():
            for vi in range(n_val_patches):
                image, target, _, _ = val_ds[vi]
                D, H, W = image.shape[1:]
                fg = np.argwhere(target[2].numpy() > 0)
                rng = np.random.default_rng(1000 + vi)
                c = fg[rng.integers(len(fg))] if len(fg) else np.array([D//2, H//2, W//2])
                st = [int(np.clip(int(cc) - p // 2, 0, max(0, full - p)))
                     for cc, p, full in zip(c, (pd_, ph_, pw_), (D, H, W))]
                sl = tuple(slice(s, min(s + p, full))
                          for s, p, full in zip(st, (pd_, ph_, pw_), (D, H, W)))
                img_c = image[(slice(None),) + sl]; tgt_c = target[(slice(None),) + sl]
                if img_c.shape[1:] != (pd_, ph_, pw_):
                    ip = torch.zeros((img_c.shape[0], pd_, ph_, pw_), dtype=img_c.dtype)
                    tp_ = torch.zeros((tgt_c.shape[0], pd_, ph_, pw_), dtype=tgt_c.dtype)
                    ip[:, :img_c.shape[1], :img_c.shape[2], :img_c.shape[3]] = img_c
                    tp_[:, :tgt_c.shape[1], :tgt_c.shape[2], :tgt_c.shape[3]] = tgt_c
                    img_c, tgt_c = ip, tp_
                img_p = img_c.unsqueeze(0).to(dev); tgt_p = tgt_c.unsqueeze(0).to(dev)
                p_ = model(img_p)['probs']
                pb = (p_ >= 0.5).float()
                for r in range(3):
                    inter = (pb[:, r] * tgt_p[:, r]).sum()
                    den = pb[:, r].sum() + tgt_p[:, r].sum()
                    d = (2*inter/den.clamp_min(1e-6)) if den > 0 else torch.tensor(1.0)
                    dices[r].append(float(d.item()))
                    fps[r] += int(((pb[:, r]==1)&(tgt_p[:, r]==0)).sum().item())

            # ---- lesion trajectory tracking (BOTH frozen sets, every epoch) ----
            for sid, lesions in lesion_sets.items():
                subj_idx = sid_to_idx.get(sid)
                if subj_idx is None:
                    continue
                image_native, target_native, _ = train_loader.dataset._load_subject(
                    train_loader.dataset.subject_dirs[subj_idx])
                D, H, W = image_native.shape[1:]
                from scipy import ndimage as ndi_
                et_lbl, et_n = ndi_.label(target_native[ET] > 0.5)
                for lesion in lesions:
                    cid = lesion['comp_id']
                    if cid < 1 or cid > et_n:
                        continue
                    cm = et_lbl == cid
                    if not cm.any():
                        continue
                    lz, ly, lx = np.where(cm)
                    l_lo = np.array([lz.min(), ly.min(), lx.min()])
                    l_hi = np.array([lz.max(), ly.max(), lx.max()])
                    pad = 16
                    z0 = max(0, int(l_lo[0]) - pad); z1 = min(D, int(l_hi[0]) + pad + 1)
                    y0 = max(0, int(l_lo[1]) - pad); y1 = min(H, int(l_hi[1]) + pad + 1)
                    x0 = max(0, int(l_lo[2]) - pad); x1 = min(W, int(l_hi[2]) + pad + 1)
                    crop_img = image_native[:, z0:z1, y0:y1, x0:x1]
                    crop_tgt_et = (target_native[ET, z0:z1, y0:y1, x0:x1] > 0.5)
                    cd, ch, cw = crop_img.shape[1:]
                    pdz = (-cd) % 16; phz = (-ch) % 16; pwz = (-cw) % 16
                    if pdz or phz or pwz:
                        crop_img = np.pad(crop_img, ((0,0),(0,pdz),(0,phz),(0,pwz)))
                        crop_tgt_et = np.pad(crop_tgt_et, ((0,pdz),(0,phz),(0,pwz)))
                    img_t = torch.from_numpy(crop_img).unsqueeze(0).to(dev)
                    out_t = model(img_t)
                    p_et = out_t['probs'][0, ET].cpu().numpy()
                    lesion_local = cm[z0:z1, y0:y1, x0:x1]
                    if pdz or phz or pwz:
                        lesion_local = np.pad(lesion_local, ((0,pdz),(0,phz),(0,pwz)))
                    mean_prob = float(p_et[lesion_local].mean())
                    max_prob = float(p_et[lesion_local].max())
                    pb_local = p_et >= 0.5
                    inter = (pb_local & lesion_local).sum()
                    denom = pb_local.sum() + lesion_local.sum()
                    l_dice = float(2*inter/denom) if denom > 0 else 1.0
                    detected = int(inter / lesion_local.sum() >= 0.5) if lesion_local.sum() > 0 else 0
                    traj_w.writerow([epoch+1, sid, cid, lesion['group'], lesion['size'],
                                     mean_prob, max_prob, l_dice, detected])
        traj_f.flush()

        row = [epoch+1, train_loss, float(np.mean(dices[0])), float(np.mean(dices[1])),
              float(np.mean(dices[2])), fps[0], time.time()-t0]
        metrics_w.writerow(row); metrics_f.flush()
        print(f'[{run_name}] ep{epoch+1}: loss={train_loss:.4f} diceET={row[2]:.3f} '
              f'fpET={fps[0]} ({time.time()-t0:.0f}s)', flush=True)

    metrics_f.close(); traj_f.close()


def main():
    dev = torch.device('cuda')
    smoke = '--smoke' in sys.argv
    global N_EPOCHS, SMOKE_SUBJECTS
    if smoke:
        N_EPOCHS = 1
        SMOKE_SUBJECTS = 8

    lesion_sets = load_lesion_sets()
    print(f'Loaded {sum(len(v) for v in lesion_sets.values())} frozen lesions '
          f'across {len(lesion_sets)} subjects', flush=True)

    run_condition('baseline', 'E228A_baseline', dev, lesion_sets)
    run_condition('conflict', 'E228A_conflict', dev, lesion_sets)
    run_condition('random', 'E228A_random', dev, lesion_sets)
    print('E228-A complete.', flush=True)


if __name__ == '__main__':
    main()
