"""
E180 shared stratification helper -- the E167-defined 110-subject "good" set,
reused VERBATIM (never redefined from E180 data), per the Stage 5.5 lock.
"""
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
E160_TABLE = (PROJECT_ROOT / "experiments" / "exp_e12_eggo_m" / "e160"
              / "E160_L_per_subject.json")


def e167_good_subject_ids():
    """The 110, defined BY ID from E160's stored table -- identical to
    run_e167_boundary_residual.py:good_subject_ids(). Never recomputed from
    E180 data, per the Stage 5.5 lock."""
    recs = json.load(open(E160_TABLE))
    det = np.array([r["dice_intact"]["ET"] for r in recs])
    dtc = np.array([r["dice_intact"]["TC"] for r in recs])
    k = 13
    bad = set(np.argsort(det)[:k].tolist()) | set(np.argsort(dtc)[:k].tolist())
    good = [recs[i]["sid"] for i in range(len(recs)) if i not in bad]
    assert len(good) == len(recs) - len(bad), "subject bookkeeping mismatch"
    return set(good)


def e160_baseline_dice(sid):
    """dice_intact ET/TC/WT for a subject, from E160's table."""
    recs = json.load(open(E160_TABLE))
    by_sid = {r["sid"]: r for r in recs}
    r = by_sid.get(sid)
    if r is None:
        return None
    return r["dice_intact"]


def hard_tail_category(sid, et_thresh=0.3, tc_thresh=0.3):
    """well_segmented / poor_et / poor_tc / poor_both / zero_et_tc, for the
    hard-tail diagnosis (Stage 5.5 lock item 4)."""
    d = e160_baseline_dice(sid)
    if d is None:
        return "unknown"
    et, tc = d["ET"], d["TC"]
    if et == 0.0 and tc == 0.0:
        return "zero_et_tc"
    poor_et = et < et_thresh
    poor_tc = tc < tc_thresh
    if poor_et and poor_tc:
        return "poor_both"
    if poor_et:
        return "poor_et"
    if poor_tc:
        return "poor_tc"
    return "well_segmented"
