"""E233 -- Hypothesis 1 test (mechanism matrix, per explicit user framing):
"the input contains insufficient discriminative information" for
persistently-missed ET lesions. NO MODEL, NO CHECKPOINT, NO TRAINING --
pure raw-MRI-intensity measurement. This is the cheapest, most decisive
test in the mechanism matrix and is run FIRST, correctly, because a
positive result here would end the search (no architecture/training
intervention can fix information that genuinely is not in the input),
and this is the ONE axis this entire session has never measured --
E226-E232 all used model activations/gradients, which can only speak to
what the NETWORK does with the input, never to what's actually IN it.

MEASURED, per GT ET component l, per modality m in {t1c,t1n,t2f,t2w}:
  lesion_mean_m   = mean raw (z-normalized-per-subject) intensity over
                    the lesion's own voxels
  shell_mean_m    = mean intensity over a surrounding SHELL (dilated
                    lesion mask minus the lesion itself, i.e. immediately
                    adjacent normal-appearing tissue -- the natural local
                    reference/comparison population a classifier would
                    need to discriminate against)
  shell_std_m     = std of intensity over that same shell
  contrast_m      = (lesion_mean_m - shell_mean_m) / (shell_std_m + eps)
                    -- a z-score-style local separability measure: how
                    many shell-standard-deviations does the lesion's own
                    mean intensity sit from its immediate surroundings,
                    per modality. This is EXACTLY the quantity a linear/
                    simple classifier would need to be large for the
                    lesion to be discriminable from local background --
                    if this is near ZERO for missed lesions (size-matched
                    against detected), the raw MRI signal itself does not
                    separate the lesion from its surroundings at that
                    location, supporting hypothesis 1.

Outcome variable: `detected` REUSED from E223_exposure.csv (same
checkpoint/MIN_VOX=5/labeling convention as this entire session) --
NOT recomputed, no model touched by this script at all.
"""
import sys, csv, time
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy import ndimage

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from Dataset.brats_multimodal_dataset import (  # noqa: E402
    BraTSMultimodalDataset, MODALITIES, znorm_brain)

MIN_VOX = 5
ET = 0
SHELL_DILATION = 3  # voxels -- immediate surrounding shell width


def main():
    smoke = '--smoke' in sys.argv

    ds = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'train',
                                val_split=0.1, patch_size=(128, 128, 128))

    # need the SAME (subject_id, comp_id) population E223/E226/E227 used,
    # for direct comparability -- reuse E223's own subject ordering by
    # simply iterating ALL train subjects (E223 covered all 1126), so
    # every (subject,comp) pair E223 measured will be found here too.
    e223_path = HERE / 'E223_exposure.csv'
    e223_keys = set()
    with open(e223_path) as f:
        for row in csv.DictReader(f):
            e223_keys.add((row['subject_id'], row['comp_id']))
    print(f'E233 intensity separability: {len(e223_keys)} lesions to match '
          f'(from E223_exposure.csv)', flush=True)

    n_total = 5 if smoke else len(ds)
    out = HERE / ('E233_smoke.csv' if smoke else 'E233_intensity.csv')
    fh = open(out, 'w', newline='')
    fieldnames = ['subject_id', 'comp_id', 'size']
    for m in MODALITIES:
        fieldnames += [f'contrast_{m}', f'lesion_mean_{m}', f'shell_mean_{m}', f'shell_std_{m}']
    w = csv.DictWriter(fh, fieldnames=fieldnames)
    w.writeheader()

    t0 = time.time()
    n_measured = 0
    struct = ndimage.generate_binary_structure(3, 1)
    for ii in range(n_total):
        image, target, sid = ds._load_subject(ds.subject_dirs[ii])
        # image: (4, D, H, W) z-normalized per MODALITIES order
        Y = target > 0.5
        et_lbl, et_n = ndimage.label(Y[ET])

        for g in range(1, et_n + 1):
            cm = et_lbl == g
            sz = int(cm.sum())
            if sz < MIN_VOX:
                continue
            if (sid, str(g)) not in e223_keys:
                continue

            dilated = ndimage.binary_dilation(cm, structure=struct, iterations=SHELL_DILATION)
            shell = dilated & (~cm)
            # also exclude any OTHER ET tissue or background (outside-brain,
            # intensity exactly 0) from the shell, so the reference
            # population is genuinely "normal-appearing local tissue"
            brain_mask = image[0] != 0  # t1c channel, 0 = outside-brain (per znorm_brain convention)
            shell = shell & brain_mask & (~Y[ET])
            if shell.sum() < MIN_VOX:
                continue  # lesion too close to brain edge / other ET tissue for a valid shell

            row = {'subject_id': sid, 'comp_id': g, 'size': sz}
            for mi, m in enumerate(MODALITIES):
                vol = image[mi]
                lesion_vals = vol[cm]
                shell_vals = vol[shell]
                lesion_mean = float(lesion_vals.mean())
                shell_mean = float(shell_vals.mean())
                shell_std = float(shell_vals.std())
                contrast = (lesion_mean - shell_mean) / (shell_std + 1e-6)
                row[f'contrast_{m}'] = contrast
                row[f'lesion_mean_{m}'] = lesion_mean
                row[f'shell_mean_{m}'] = shell_mean
                row[f'shell_std_{m}'] = shell_std
            w.writerow(row)
            n_measured += 1
        fh.flush()
        if (ii + 1) % 100 == 0 or smoke:
            print(f'  {ii+1}/{n_total} subj, {n_measured} lesions measured '
                  f'({time.time()-t0:.0f}s)', flush=True)
    fh.close()
    print(f'wrote {out.name}: {n_measured} lesions ({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
