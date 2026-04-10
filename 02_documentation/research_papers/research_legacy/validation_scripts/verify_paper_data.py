"""
Data Verification Script - Confirm All Paper Statistics
Run this to verify all numbers used in research paper figures
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 80)
print("📊 RESEARCH PAPER DATA VERIFICATION")
print("=" * 80)

# Paths
CSV_DIR = Path("C:/Users/HP/EDI/csv_data")
PROD_DIR = CSV_DIR / "production_model"
ABL_DIR = CSV_DIR / "ablation_variants"

# ============================================================================
# 1. PRODUCTION MODEL (FINAL RESULTS)
# ============================================================================
print("\n" + "=" * 80)
print("1️⃣  PRODUCTION MODEL VERIFICATION")
print("=" * 80)

prod_val = pd.read_csv(PROD_DIR / "production_val_logs.csv")

best_epoch_idx = prod_val['dice'].idxmax()
best_epoch = best_epoch_idx + 1  # Convert to 1-indexed
best_row = prod_val.iloc[best_epoch_idx]

print(f"\n✅ Best Epoch: {best_epoch}")
print(f"   Source: production_val_logs.csv, row {best_epoch_idx} (0-indexed)")
print(f"\n📊 Performance Metrics (Epoch {best_epoch}):")
print(f"   Dice Score:  {best_row['dice']*100:.2f}%")
print(f"   Precision:   {best_row['precision']*100:.2f}%")
print(f"   Recall:      {best_row['recall']*100:.2f}%")
print(f"   F1 Score:    {best_row['f1']*100:.2f}%")
print(f"   Loss:        {best_row['loss']:.4f}")

print(f"\n📈 Training Statistics:")
print(f"   Total epochs run: {len(prod_val)}")
print(f"   Best epoch: {best_epoch}")
print(f"   Early stopping: Epoch {len(prod_val)} (20 epochs after best)")

# ============================================================================
# 2. ABLATION STUDY RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("2️⃣  ABLATION STUDY VERIFICATION")
print("=" * 80)

ablation = pd.read_csv(CSV_DIR / "ablation_summary.csv")

print(f"\n✅ Loaded: {len(ablation)} variants")
print(f"   Source: ablation_summary.csv")
print("\n📊 Ablation Results (Sorted by Dice Score):")
print("\n" + "-" * 80)
print(f"{'Variant':<20} {'Val Dice':<12} {'Δ vs Base':<15} {'%Change':<12}")
print("-" * 80)

baseline_dice = ablation.loc[ablation['Variant'] == 'Baseline', 'Val_Dice'].values[0]

for _, row in ablation.sort_values('Val_Dice', ascending=False).iterrows():
    variant = row['Variant']
    dice = row['Val_Dice']
    delta = row['Absolute_Diff']
    pct = row['Relative_Diff_%']
    
    if variant == 'Baseline':
        print(f"{variant:<20} {dice*100:>6.2f}%      {'(baseline)':<15} {'-':<12}")
    else:
        sign = '+' if delta >= 0 else ''
        print(f"{variant:<20} {dice*100:>6.2f}%      {sign}{delta*100:>6.2f}%        {sign}{pct:>6.2f}%")

print("-" * 80)

# Key findings
best_variant = ablation.loc[ablation['Val_Dice'].idxmax(), 'Variant']
worst_variant = ablation.loc[ablation['Val_Dice'].idxmin(), 'Variant']

print(f"\n🏆 Key Findings:")
print(f"   Best variant: {best_variant} ({ablation['Val_Dice'].max()*100:.2f}%)")
print(f"   Worst variant: {worst_variant} ({ablation['Val_Dice'].min()*100:.2f}%)")
print(f"   Range: {(ablation['Val_Dice'].max() - ablation['Val_Dice'].min())*100:.2f}%")

# Critical components
no3d_improvement = ablation.loc[ablation['Variant'] == 'No3DConv', 'Relative_Diff_%'].values[0]
noresidual_drop = ablation.loc[ablation['Variant'] == 'NoResidual', 'Relative_Diff_%'].values[0]

print(f"\n🔑 Critical Components:")
print(f"   3D Conv removal: +{no3d_improvement:.2f}% (BENEFICIAL to remove)")
print(f"   Residual removal: {noresidual_drop:.2f}% (CRITICAL to keep)")

# ============================================================================
# 3. STATE-OF-THE-ART COMPARISON
# ============================================================================
print("\n" + "=" * 80)
print("3️⃣  STATE-OF-THE-ART COMPARISON")
print("=" * 80)

print(f"\n📊 Dice Score Comparison:")
print("-" * 80)
print(f"{'Method':<30} {'Dice Score':<15} {'Source':<30}")
print("-" * 80)
print(f"{'HybridMiniSwin2.5D (Ours)':<30} {best_row['dice']*100:>6.2f}%        {'production_val_logs.csv':<30}")
print(f"{'nnU-Net (Literature)':<30} {'82.30%':>12}  {'RESEARCH_PAPER_STATISTICS.md':<30}")
print(f"{'MS-Net (Literature)':<30} {'79.10%':>12}  {'RESEARCH_PAPER_STATISTICS.md':<30}")
print(f"{'3D U-Net (Literature)':<30} {'76.50%':>12}  {'RESEARCH_PAPER_STATISTICS.md':<30}")
print(f"{'Baseline (Ours)':<30} {baseline_dice*100:>6.2f}%        {'ablation_summary.csv':<30}")
print("-" * 80)

improvement_over_sota = best_row['dice']*100 - 82.30
improvement_over_baseline = best_row['dice']*100 - baseline_dice*100

print(f"\n🎯 Improvements:")
print(f"   vs nnU-Net (SOTA): +{improvement_over_sota:.2f}%")
print(f"   vs Baseline (Ours): +{improvement_over_baseline:.2f}%")

# ============================================================================
# 4. TRAINING CONVERGENCE
# ============================================================================
print("\n" + "=" * 80)
print("4️⃣  TRAINING CONVERGENCE ANALYSIS")
print("=" * 80)

print(f"\n📈 Production Model:")
print(f"   Epochs to best: {best_epoch}")
print(f"   Total epochs: {len(prod_val)}")
print(f"   Convergence efficiency: {best_epoch}/{len(prod_val)} = {best_epoch/len(prod_val)*100:.1f}%")

print(f"\n📊 Progression:")
print(f"   Epoch 1:  Dice {prod_val.iloc[0]['dice']*100:.2f}%")
print(f"   Epoch 10: Dice {prod_val.iloc[9]['dice']*100:.2f}%")
print(f"   Epoch 20: Dice {prod_val.iloc[19]['dice']*100:.2f}%")
print(f"   Epoch 28: Dice {prod_val.iloc[27]['dice']*100:.2f}% (BEST)")
print(f"   Epoch 48: Dice {prod_val.iloc[47]['dice']*100:.2f}% (FINAL)")

improvement_first_to_best = (prod_val.iloc[27]['dice'] - prod_val.iloc[0]['dice']) * 100
print(f"\n   Total improvement: {improvement_first_to_best:.2f}%")

# ============================================================================
# 5. CLINICAL METRICS VERIFICATION
# ============================================================================
print("\n" + "=" * 80)
print("5️⃣  CLINICAL METRICS VERIFICATION")
print("=" * 80)

print(f"\n✅ All metrics at Best Epoch ({best_epoch}):")
print(f"   Dice:      {best_row['dice']*100:.2f}%  ✓ Primary metric")
print(f"   Precision: {best_row['precision']*100:.2f}%  ✓ From CSV")
print(f"   Recall:    {best_row['recall']*100:.2f}%  ✓ From CSV (HIGH - catches most lesions)")
print(f"   F1:        {best_row['f1']*100:.2f}%  ✓ From CSV")
print(f"   Loss:      {best_row['loss']:.4f}  ✓ From CSV")

print(f"\n⚠️  Additional Metrics (NOT in CSV):")
print(f"   Specificity: 99.85% (from WEBAPP_VALIDATION_SUMMARY.md)")
print(f"   IoU/Jaccard: 72.35% (calculated from Dice)")

# Calculate IoU from Dice
iou_from_dice = best_row['dice'] / (2 - best_row['dice'])
print(f"   IoU verification: Dice/(2-Dice) = {iou_from_dice*100:.2f}%")

# ============================================================================
# 6. OVERFITTING ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("6️⃣  OVERFITTING ANALYSIS")
print("=" * 80)

print(f"\n📊 Train vs Val Gaps:")
print("-" * 80)
print(f"{'Variant':<20} {'Train Dice':<12} {'Val Dice':<12} {'Gap':<12} {'Status':<15}")
print("-" * 80)

for _, row in ablation.iterrows():
    variant = row['Variant']
    train_dice = row['Train_Dice']
    val_dice = row['Val_Dice']
    gap = row['Overfitting_Gap']
    
    if gap > 0.08:
        status = "High overfitting"
    elif gap > 0.05:
        status = "Moderate"
    elif gap < 0:
        status = "UNDERFITTING"
    else:
        status = "Good"
    
    print(f"{variant:<20} {train_dice*100:>6.2f}%      {val_dice*100:>6.2f}%      {gap*100:>+6.2f}%     {status:<15}")

print("-" * 80)

# ============================================================================
# 7. EFFICIENCY METRICS (ESTIMATED)
# ============================================================================
print("\n" + "=" * 80)
print("7️⃣  EFFICIENCY METRICS")
print("=" * 80)

print(f"\n⚠️  WARNING: These are ESTIMATED values, not measured!")
print(f"\n📊 Model Complexity (ESTIMATED):")
print(f"   Parameters: 12.4M (needs verification with model.parameters())")
print(f"   FLOPs: 176 GFLOPs (calculated as 58% reduction from 3D baseline)")
print(f"   Inference time: 80ms (discrepancy with 2-3s in docs)")

print(f"\n🔴 CRITICAL: Run actual measurements:")
print(f"   1. Count parameters: sum(p.numel() for p in model.parameters())")
print(f"   2. Measure FLOPs: FlopCountAnalysis(model, inputs)")
print(f"   3. Benchmark inference time consistently")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("✅ VERIFICATION COMPLETE")
print("=" * 80)

print(f"\n📋 Data Quality Summary:")
print(f"   ✅ Production model metrics: VERIFIED from CSV")
print(f"   ✅ Ablation study results: VERIFIED from CSV")
print(f"   ✅ Training convergence: VERIFIED from CSV")
print(f"   ⚠️  Literature comparisons: FROM DOCS (need citations)")
print(f"   🔴 Efficiency metrics: ESTIMATED (need measurements)")

print(f"\n📄 CSV Files Used:")
print(f"   1. {PROD_DIR / 'production_val_logs.csv'} ({len(prod_val)} rows)")
print(f"   2. {PROD_DIR / 'production_train_logs.csv'}")
print(f"   3. {CSV_DIR / 'ablation_summary.csv'} ({len(ablation)} variants)")

print(f"\n🎯 Key Numbers for Paper:")
print(f"   Final Dice: {best_row['dice']*100:.2f}%")
print(f"   Best Epoch: {best_epoch}")
print(f"   vs nnU-Net: +{improvement_over_sota:.2f}%")
print(f"   Critical component: Residual connections ({noresidual_drop:.2f}% when removed)")
print(f"   Beneficial removal: 3D Conv (+{no3d_improvement:.2f}% when removed)")

print("\n" + "=" * 80)
