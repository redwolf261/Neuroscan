import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESEARCH_INFRA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

adaptive = pd.read_csv(os.path.join(RESEARCH_INFRA, "reproduction_attempt_adaptive", "segmentation_log.csv"))
center = pd.read_csv(os.path.join(RESEARCH_INFRA, "reproduction_attempt_center_window", "segmentation_log.csv"))
multi = pd.read_csv(os.path.join(RESEARCH_INFRA, "reproduction_attempt_multislice", "segmentation_log.csv"))

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
ax.plot(adaptive['epoch'], adaptive['val_dice'], label=f"Adaptive (broken) — best {adaptive['val_dice'].max():.5f}",
        color='tab:blue', linewidth=1.3, marker='o', markersize=2)
ax.plot(center['epoch'], center['val_dice'], label=f"Center-window fix — best {center['val_dice'].max():.5f}",
        color='tab:red', linewidth=1.3, marker='o', markersize=2)
ax.plot(multi['epoch'], multi['val_dice_fixed32'], label=f"Multi-slice, evaluated @ fixed idx 32 — best {multi['val_dice_fixed32'].max():.5f}",
        color='tab:green', linewidth=1.3, marker='o', markersize=2)
ax.set_xlabel("Epoch")
ax.set_ylabel("Validation Dice")
ax.set_title("Same metric as before: Dice at fixed center slice (idx 32)\n(apples-to-apples with earlier reproduction attempts)")
ax.legend(fontsize=8)

ax = axes[1]
ax.plot(multi['epoch'], multi['val_dice_fixed32'], label=f"Multi-slice @ fixed idx 32 — best {multi['val_dice_fixed32'].max():.5f}",
        color='tab:green', linewidth=1.3, marker='o', markersize=2)
ax.plot(multi['epoch'], multi['val_dice_random'], label=f"Multi-slice @ random informative slices — best {multi['val_dice_random'].max():.5f}",
        color='darkgreen', linewidth=1.8, marker='o', markersize=2)
ax.axhline(adaptive['val_dice'].max(), color='tab:blue', linestyle='--', linewidth=1,
           label=f"Adaptive best (fixed@32): {adaptive['val_dice'].max():.5f}")
ax.set_xlabel("Epoch")
ax.set_ylabel("Validation Dice")
ax.set_title("Multi-slice model: two different evaluation protocols\n(same model, same epoch, different metric)")
ax.legend(fontsize=8)

fig.suptitle("Reproduction attempt: 9-patient dataset, real early stopping, matched budgets", y=1.02)
fig.tight_layout()
out_path = os.path.join(RESEARCH_INFRA, "phase8_figures", "three_way_reproduction_comparison.png")
fig.savefig(out_path, dpi=200, bbox_inches='tight')
print(f"wrote {out_path}")

print("\n=== SUMMARY TABLE ===")
print(f"{'Approach':40s} {'Best Dice (fixed@32 metric)':>28s} {'Best Dice (native/random metric)':>34s}")
print(f"{'Adaptive (as-shipped, broken)':40s} {adaptive['val_dice'].max():>28.5f} {'n/a (only ever evaluated @32)':>34s}")
print(f"{'Center-window fix':40s} {center['val_dice'].max():>28.5f} {'n/a (only ever evaluated @32)':>34s}")
print(f"{'Multi-slice supervision':40s} {multi['val_dice_fixed32'].max():>28.5f} {multi['val_dice_random'].max():>34.5f}")
