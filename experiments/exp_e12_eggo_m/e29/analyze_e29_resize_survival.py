import json
from pathlib import Path
import numpy as np
from scipy import stats

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False

BASE = Path(__file__).parent
FIG_DIR = BASE / "figures"
FIG_DIR.mkdir(exist_ok=True)

with open(BASE / "E29_resize_survival_table.json") as f:
    records = json.load(f)
n = len(records)
print(f"Loaded {n} native-space GT components")

native_size = np.array([r["native_size"] for r in records])
size_64 = np.array([r["size_at_64"] for r in records])
size_96 = np.array([r["size_at_96"] for r in records])
size_128 = np.array([r["size_at_128"] for r in records])

print("\n=== Overall survival rates ===")
for res, sizes in [(64, size_64), (96, size_96), (128, size_128)]:
    survive = (sizes > 0).mean()
    print(f"  {res}^3: {100*survive:.1f}% of native components survive (>0 voxels), mean size={sizes.mean():.2f}, median={np.median(sizes):.1f}")

print("\n=== Native size distribution ===")
print(f"  mean={native_size.mean():.1f} median={np.median(native_size):.1f} min={native_size.min()} max={native_size.max()}")
pct = np.percentile(native_size, [5,25,50,75,95])
print(f"  p5={pct[0]:.1f} p25={pct[1]:.1f} p50={pct[2]:.1f} p75={pct[3]:.1f} p95={pct[4]:.1f}")

# ============================================================
# Native size bins -- how does resize behave for SMALL native lesions specifically?
# ============================================================
print("\n=== Resize behavior by native-size bin ===")
native_bins = [(1, 10, "1-10"), (10, 50, "10-50"), (50, 150, "50-150"), (150, 500, "150-500"), (500, np.inf, ">500")]
bin_results = {}
for lo, hi, lab in native_bins:
    mask = (native_size > lo) & (native_size <= hi) if lo > 1 else (native_size >= lo) & (native_size <= hi)
    nn = mask.sum()
    if nn == 0:
        continue
    row = {"n": int(nn)}
    print(f"  native size {lab} (n={nn}):")
    for res, sizes in [(64, size_64), (96, size_96), (128, size_128)]:
        s = sizes[mask]
        survive_rate = (s > 0).mean()
        mean_s = s.mean()
        median_s = np.median(s)
        print(f"    {res}^3: survive={100*survive_rate:.1f}%  mean_voxels={mean_s:.2f}  median_voxels={median_s:.1f}")
        row[f"survive_{res}"] = float(survive_rate)
        row[f"mean_voxels_{res}"] = float(mean_s)
        row[f"median_voxels_{res}"] = float(median_s)
    bin_results[lab] = row

# ============================================================
# The key plot data: P(survives) vs native size, for each resolution
# ============================================================
print("\n=== P(component becomes <=3 voxels, i.e. 'near-vanishing') by native size and resolution ===")
near_vanish_results = {}
for lo, hi, lab in native_bins:
    mask = (native_size > lo) & (native_size <= hi) if lo > 1 else (native_size >= lo) & (native_size <= hi)
    nn = mask.sum()
    if nn == 0:
        continue
    row = {"n": int(nn)}
    for res, sizes in [(64, size_64), (96, size_96), (128, size_128)]:
        near_vanish = ((sizes[mask] >= 1) & (sizes[mask] <= 3)).mean()
        zero = (sizes[mask] == 0).mean()
        row[f"pct_vanished_{res}"] = float(zero)
        row[f"pct_near_vanish_1to3_{res}"] = float(near_vanish)
    near_vanish_results[lab] = row
    print(f"  native {lab}: " + "  ".join(f"{res}^3: vanished={100*row[f'pct_vanished_{res}']:.1f}% near-vanish(1-3vox)={100*row[f'pct_near_vanish_1to3_{res}']:.1f}%" for res in (64,96,128)))

# ============================================================
# Direct comparison: how much does going from 64 to 128 change small lesions specifically?
# ============================================================
print("\n=== Direct 64->128 voxel-count gain for native lesions <=50 voxels ===")
small_mask = (native_size >= 1) & (native_size <= 50)
gain_64_to_128 = size_128[small_mask] - size_64[small_mask]
ratio_128_over_64 = np.divide(size_128[small_mask], np.maximum(size_64[small_mask], 1e-9))
print(f"  n={small_mask.sum()}")
print(f"  mean voxel gain (128 - 64): {gain_64_to_128.mean():.2f}")
print(f"  median voxel gain: {np.median(gain_64_to_128):.2f}")
print(f"  mean ratio (128/64, where 64>0): {np.mean(ratio_128_over_64[size_64[small_mask]>0]):.2f}x" if (size_64[small_mask]>0).any() else "  n/a (all vanish at 64)")

# how many go from "vanished/near-vanished at 64" to "meaningfully present at 128"?
vanished_at_64 = size_64[small_mask] <= 3
present_at_128 = size_128[small_mask] >= 5
rescued = (vanished_at_64 & present_at_128).sum()
print(f"  components with size<=3 at 64^3 that reach >=5 voxels at 128^3: {rescued} / {vanished_at_64.sum()} ({100*rescued/max(vanished_at_64.sum(),1):.1f}% of those that were near-vanished at 64)")

results = {
    "n_components": n,
    "overall_survival": {
        res: {"survival_rate": float((sizes>0).mean()), "mean_size": float(sizes.mean()), "median_size": float(np.median(sizes))}
        for res, sizes in [(64, size_64), (96, size_96), (128, size_128)]
    },
    "native_size_distribution": {
        "mean": float(native_size.mean()), "median": float(np.median(native_size)),
        "min": int(native_size.min()), "max": int(native_size.max()),
        "p5": float(pct[0]), "p25": float(pct[1]), "p50": float(pct[2]), "p75": float(pct[3]), "p95": float(pct[4]),
    },
    "resize_by_native_bin": bin_results,
    "near_vanish_by_bin": near_vanish_results,
    "small_lesion_64_to_128_gain": {
        "n": int(small_mask.sum()),
        "mean_voxel_gain": float(gain_64_to_128.mean()),
        "median_voxel_gain": float(np.median(gain_64_to_128)),
        "n_rescued_from_near_vanish": int(rescued),
        "n_near_vanished_at_64": int(vanished_at_64.sum()),
    },
}
with open(BASE / "E29_resize_survival_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved to {BASE / 'E29_resize_survival_results.json'}")

# ============================================================
# THE key plot: P(survives at each resolution) vs native size
# ============================================================
if HAVE_MPL:
    order = np.argsort(native_size)
    ns_sorted = native_size[order]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for res, sizes, color in [(64, size_64, "#c44e52"), (96, size_96, "#dd8452"), (128, size_128, "#55a868")]:
        s_sorted = sizes[order]
        axes[0].scatter(ns_sorted, s_sorted, s=6, alpha=0.35, color=color, label=f"{res}³")
    axes[0].plot([1, native_size.max()], [1, native_size.max()], "k--", linewidth=0.8, label="y=x (no change)")
    axes[0].set_xscale("log")
    axes[0].set_yscale("symlog", linthresh=1)
    axes[0].set_xlabel("Native lesion size (voxels)")
    axes[0].set_ylabel("Resized lesion size (voxels)")
    axes[0].set_title("Figure 1: Native vs resized component size")
    axes[0].legend()

    bin_labels = [lab for lo, hi, lab in native_bins if lab in near_vanish_results]
    x = np.arange(len(bin_labels))
    width = 0.25
    for i, res in enumerate((64, 96, 128)):
        vals = [100*near_vanish_results[lab][f"pct_vanished_{res}"] for lab in bin_labels]
        axes[1].bar(x + (i-1)*width, vals, width, label=f"{res}³")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(bin_labels, rotation=30)
    axes[1].set_xlabel("Native lesion size bin (voxels)")
    axes[1].set_ylabel("% components fully vanished (0 voxels)")
    axes[1].set_title("Figure 2: Vanishing rate by native size and resolution")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "e29_resize_survival_figures.png", dpi=120)
    plt.close()
    print("Saved figures")
