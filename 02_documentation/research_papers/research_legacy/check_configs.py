"""
Check which hyperparameter configs are completed and which are pending
"""
import os
import pandas as pd

# Configuration
K_SLICES_VALUES = [3, 5, 7, 9]
WINDOW_SIZE_VALUES = [4, 8, 16]
EPOCHS = 50

GDRIVE_BASE = r"G:\My Drive\NeuroScan_Research\Research_HyperparamSensitivity"

print("=" * 80)
print("CHECKING HYPERPARAMETER SENSITIVITY CONFIGS")
print("=" * 80)

completed = []
incomplete = []
not_started = []

for k_slices in K_SLICES_VALUES:
    for window_size in WINDOW_SIZE_VALUES:
        config_name = f"k{k_slices}_w{window_size}"
        config_dir = os.path.join(GDRIVE_BASE, config_name)
        val_log_path = os.path.join(config_dir, "val_logs.csv")
        
        if os.path.exists(val_log_path):
            try:
                df = pd.read_csv(val_log_path)
                epochs_done = len(df)
                
                if epochs_done >= EPOCHS:
                    # Get best dice
                    best_dice = df['dice'].max()
                    completed.append((config_name, epochs_done, best_dice))
                else:
                    incomplete.append((config_name, epochs_done))
            except Exception as e:
                not_started.append((config_name, f"Error: {e}"))
        else:
            not_started.append((config_name, "No val_logs.csv"))

print(f"\n✅ COMPLETED ({len(completed)}):")
for config_name, epochs, best_dice in completed:
    print(f"   {config_name}: {epochs} epochs, Best Dice: {best_dice:.4f}")

print(f"\n⚠️  INCOMPLETE ({len(incomplete)}):")
for config_name, epochs in incomplete:
    print(f"   {config_name}: {epochs}/{EPOCHS} epochs")

print(f"\n❌ NOT STARTED ({len(not_started)}):")
for config_name, reason in not_started:
    print(f"   {config_name}: {reason}")

print(f"\n" + "=" * 80)
print(f"SUMMARY:")
print(f"  Completed: {len(completed)}/12")
print(f"  Incomplete: {len(incomplete)}/12")
print(f"  Not Started: {len(not_started)}/12")
print(f"  Total Progress: {len(completed)}/{len(K_SLICES_VALUES) * len(WINDOW_SIZE_VALUES)}")
print("=" * 80)
