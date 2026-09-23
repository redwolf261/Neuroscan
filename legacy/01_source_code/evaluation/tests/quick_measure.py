import torch

print("=" * 80)
print("MEASURING COMPUTATIONAL EFFICIENCY FROM CHECKPOINTS")
print("=" * 80)

# Trial model
print("\n🔴 TRIAL MODEL:")
trial_path = r"G:\My Drive\NeuroScan_PEDiMS_v2\checkpoints\best_model.pth"
trial = torch.load(trial_path, map_location='cpu')
# Trial checkpoint is just the state dict
if isinstance(trial, dict) and "model_state_dict" in trial:
    trial_state = trial["model_state_dict"]
else:
    trial_state = trial
trial_params = sum(p.numel() for p in trial_state.values())
print(f"   Parameters: {trial_params:,} ({trial_params/1e6:.2f}M)")
print(f"   Checkpoint Size: 3.14 MB")

# Final model
print("\n🟢 FINAL MODEL:")
final_path = r"G:\My Drive\NeuroScan_FinalModel_2.5D_MAE\segmentation\best_model.pth"
final = torch.load(final_path, map_location='cpu')
# Final checkpoint has model_state_dict key
if isinstance(final, dict) and "model_state_dict" in final:
    final_state = final["model_state_dict"]
else:
    final_state = final
final_params = sum(p.numel() for p in final_state.values())
print(f"   Parameters: {final_params:,} ({final_params/1e6:.2f}M)")
print(f"   Checkpoint Size: 366.91 MB")

# Comparison
print("\n📊 COMPARISON:")
print(f"   Parameter Ratio: {final_params/trial_params:.2f}×")
print(f"   Size Ratio: {366.91/3.14:.2f}×")
print(f"   Parameter Difference: +{(final_params-trial_params)/1e6:.2f}M")

print("\n" + "=" * 80)
