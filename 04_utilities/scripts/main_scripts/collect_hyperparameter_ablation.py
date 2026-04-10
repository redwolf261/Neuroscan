"""
Collect all hyperparameter ablation results (K_SLICES × WINDOW_SIZE)
12 configurations total: k∈{3,5,7,9} × window∈{4,8,16}
"""

import os
import pandas as pd
import numpy as np

base_path = r"G:\My Drive\NeuroScan_Research\Research_HyperparamSensitivity"
output_path = r"C:\Users\HP\EDI\ablation_results\hyperparameter_ablation_summary.csv"

configs = [
    ('k3_w4', 3, 4),
    ('k3_w8', 3, 8),
    ('k3_w16', 3, 16),
    ('k5_w4', 5, 4),
    ('k5_w8', 5, 8),
    ('k5_w16', 5, 16),
    ('k7_w4', 7, 4),
    ('k7_w8', 7, 8),
    ('k7_w16', 7, 16),
    ('k9_w4', 9, 4),
    ('k9_w8', 9, 8),
    ('k9_w16', 9, 16),
]

results = []

print("=" * 80)
print("COLLECTING HYPERPARAMETER ABLATION RESULTS")
print("=" * 80)

for config_name, k_slices, window_size in configs:
    val_logs_path = os.path.join(base_path, config_name, "val_logs.csv")
    
    if not os.path.exists(val_logs_path):
        print(f"⚠️ Missing: {config_name}")
        continue
    
    # Load validation logs
    df = pd.read_csv(val_logs_path)
    
    # Find best epoch (highest Dice)
    best_idx = df['dice'].idxmax()
    best_row = df.loc[best_idx]
    
    result = {
        'config': config_name,
        'k_slices': k_slices,
        'window_size': window_size,
        'best_epoch': int(best_row['epoch']),
        'best_dice': float(best_row['dice']),
        'best_precision': float(best_row['precision']),
        'best_recall': float(best_row['recall']),
        'best_f1': float(best_row['f1']),
        'best_loss': float(best_row['loss'])
    }
    
    results.append(result)
    print(f"✅ {config_name}: Dice {result['best_dice']:.4f} (epoch {result['best_epoch']})")

# Create DataFrame
df_results = pd.DataFrame(results)
df_results = df_results.sort_values(['k_slices', 'window_size'])

# Identify baseline (k=5, window=4 - current config in final_model.py)
baseline_dice = df_results[(df_results['k_slices'] == 5) & (df_results['window_size'] == 4)]['best_dice'].values[0]

# Compute deltas
df_results['delta_dice'] = df_results['best_dice'] - baseline_dice
df_results['improvement_pct'] = (df_results['delta_dice'] / baseline_dice) * 100

# Find best configuration
best_config = df_results.loc[df_results['best_dice'].idxmax()]

print("\n" + "=" * 80)
print("HYPERPARAMETER ABLATION SUMMARY")
print("=" * 80)
print(f"\nBaseline: k=5, window=4 → Dice {baseline_dice:.4f}")
print(f"\nBest: {best_config['config']} (k={int(best_config['k_slices'])}, window={int(best_config['window_size'])}) → Dice {best_config['best_dice']:.4f} (+{best_config['improvement_pct']:.2f}%)")

print("\n" + "-" * 80)
print("All Configurations:")
print("-" * 80)
print(df_results[['config', 'k_slices', 'window_size', 'best_dice', 'delta_dice', 'improvement_pct']].to_string(index=False))
print("-" * 80)

# Save
df_results.to_csv(output_path, index=False)
print(f"\n✅ Saved to: {output_path}")

# Summary statistics
print("\n" + "=" * 80)
print("KEY FINDINGS")
print("=" * 80)

# Best k_slices (averaged over window_size)
k_avg = df_results.groupby('k_slices')['best_dice'].mean()
print(f"\n📊 Average Dice by K_SLICES:")
for k, dice in k_avg.items():
    delta = dice - baseline_dice
    print(f"   k={int(k)}: {dice:.4f} ({delta:+.4f}, {(delta/baseline_dice)*100:+.2f}%)")

best_k = k_avg.idxmax()
print(f"   → Best: k={int(best_k)} (avg Dice {k_avg[best_k]:.4f})")

# Best window_size (averaged over k_slices)
w_avg = df_results.groupby('window_size')['best_dice'].mean()
print(f"\n📊 Average Dice by WINDOW_SIZE:")
for w, dice in w_avg.items():
    delta = dice - baseline_dice
    print(f"   window={int(w)}: {dice:.4f} ({delta:+.4f}, {(delta/baseline_dice)*100:+.2f}%)")

best_w = w_avg.idxmax()
print(f"   → Best: window={int(best_w)} (avg Dice {w_avg[best_w]:.4f})")

print("\n" + "=" * 80)
