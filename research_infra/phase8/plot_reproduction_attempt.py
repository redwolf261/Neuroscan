import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESEARCH_INFRA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

adaptive = pd.read_csv(os.path.join(RESEARCH_INFRA, "reproduction_attempt_adaptive", "segmentation_log.csv"))
center = pd.read_csv(os.path.join(RESEARCH_INFRA, "reproduction_attempt_center_window", "segmentation_log.csv"))

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(adaptive['epoch'], adaptive['val_dice'], label=f"Adaptive (as-shipped) — best {adaptive['val_dice'].max():.5f}",
        color='tab:blue', linewidth=1.5, marker='o', markersize=3)
ax.plot(center['epoch'], center['val_dice'], label=f"Center-window fix — best {center['val_dice'].max():.5f}",
        color='tab:red', linewidth=1.5, marker='o', markersize=3)
ax.set_xlabel("Epoch")
ax.set_ylabel("Validation Dice")
ax.set_title("Reproduction attempt: adaptive selector vs. center-window fix\n(fresh MAE + segmentation, real early stopping, 9-patient dataset)")
ax.legend()
fig.tight_layout()
out_path = os.path.join(RESEARCH_INFRA, "phase8_figures", "reproduction_attempt_comparison.png")
fig.savefig(out_path, dpi=200)
print(f"wrote {out_path}")
print(f"adaptive best: {adaptive['val_dice'].max():.5f} at epoch {adaptive.loc[adaptive['val_dice'].idxmax(), 'epoch']}")
print(f"center_window best: {center['val_dice'].max():.5f} at epoch {center.loc[center['val_dice'].idxmax(), 'epoch']}")
