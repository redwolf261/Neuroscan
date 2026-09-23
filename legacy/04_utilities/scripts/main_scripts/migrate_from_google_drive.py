"""
Migrate all results, CSVs, and important files from Google Drive to local EDI folder
This will free up Google Drive space while preserving all experiment data locally
"""

import os
import shutil
from pathlib import Path
from datetime import datetime

# Paths
GDRIVE_BASE = r"G:\My Drive"
EDI_BASE = r"C:\Users\HP\EDI"
BACKUP_DIR = os.path.join(EDI_BASE, "google_drive_backup")
LOG_FILE = os.path.join(EDI_BASE, "migration_log.txt")

# Create backup directory with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
BACKUP_ROOT = os.path.join(BACKUP_DIR, f"migration_{timestamp}")
os.makedirs(BACKUP_ROOT, exist_ok=True)

def log(message):
    """Log to both console and file"""
    print(message)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")

def get_folder_size(folder_path):
    """Calculate folder size in MB"""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(folder_path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
    except Exception as e:
        log(f"   ⚠️ Error calculating size: {e}")
    return total_size / (1024 * 1024)  # Convert to MB

def copy_folder(src, dst, description):
    """Copy entire folder with progress"""
    if not os.path.exists(src):
        log(f"⚠️ SKIP: {description} - Source not found: {src}")
        return False
    
    size_mb = get_folder_size(src)
    log(f"\n📂 Copying: {description} ({size_mb:.1f} MB)")
    log(f"   From: {src}")
    log(f"   To: {dst}")
    
    try:
        if os.path.exists(dst):
            log(f"   ⚠️ Destination exists, backing up...")
            backup_dst = dst + "_old"
            if os.path.exists(backup_dst):
                shutil.rmtree(backup_dst)
            shutil.move(dst, backup_dst)
        
        shutil.copytree(src, dst)
        log(f"   ✅ Copied successfully!")
        return True
    except Exception as e:
        log(f"   ❌ ERROR: {e}")
        return False

def copy_file(src, dst, description):
    """Copy single file"""
    if not os.path.exists(src):
        log(f"⚠️ SKIP: {description} - Source not found")
        return False
    
    size_mb = os.path.getsize(src) / (1024 * 1024)
    log(f"\n📄 Copying: {description} ({size_mb:.2f} MB)")
    
    try:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        log(f"   ✅ Copied successfully!")
        return True
    except Exception as e:
        log(f"   ❌ ERROR: {e}")
        return False

def find_and_copy_csvs(src_folder, dst_folder, description):
    """Recursively find and copy all CSV files"""
    if not os.path.exists(src_folder):
        log(f"⚠️ SKIP: {description} - Source not found: {src_folder}")
        return 0
    
    log(f"\n📊 Searching for CSVs in: {description}")
    csv_count = 0
    
    for root, dirs, files in os.walk(src_folder):
        for file in files:
            if file.endswith('.csv'):
                src_path = os.path.join(root, file)
                rel_path = os.path.relpath(src_path, src_folder)
                dst_path = os.path.join(dst_folder, rel_path)
                
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                
                try:
                    shutil.copy2(src_path, dst_path)
                    size_kb = os.path.getsize(src_path) / 1024
                    log(f"   ✅ {rel_path} ({size_kb:.1f} KB)")
                    csv_count += 1
                except Exception as e:
                    log(f"   ❌ ERROR copying {rel_path}: {e}")
    
    log(f"   📊 Total CSVs copied: {csv_count}")
    return csv_count

# ===========================================================================================
# MAIN MIGRATION
# ===========================================================================================

log("=" * 80)
log("GOOGLE DRIVE TO LOCAL MIGRATION - START")
log("=" * 80)
log(f"Source: {GDRIVE_BASE}")
log(f"Destination: {EDI_BASE}")
log(f"Backup: {BACKUP_ROOT}")
log("=" * 80)

total_copied = 0
total_size_mb = 0

# ===========================================================================================
# 1. ABLATION RESULTS (HIGHEST PRIORITY)
# ===========================================================================================

log("\n" + "=" * 80)
log("SECTION 1: ABLATION RESULTS")
log("=" * 80)

# USALD Ablation configs (7 folders)
ablation_configs = [
    "USALD_Ablation_baseline",
    "USALD_Ablation_evidential",
    "USALD_Ablation_causal",
    "USALD_Ablation_self_correction",
    "USALD_Ablation_consistency",
    "USALD_Ablation_fdr",
    "USALD_CausalSelfCorrection"  # Full config
]

for config in ablation_configs:
    src = os.path.join(GDRIVE_BASE, config)
    dst = os.path.join(BACKUP_ROOT, "ablation_experiments", config)
    
    # Only copy CSV files and logs (not checkpoints to save space)
    if os.path.exists(src):
        csv_count = find_and_copy_csvs(
            os.path.join(src, "segmentation"),
            os.path.join(dst, "segmentation"),
            f"{config} - Segmentation CSVs"
        )
        csv_count += find_and_copy_csvs(
            os.path.join(src, "mae_pretraining"),
            os.path.join(dst, "mae_pretraining"),
            f"{config} - MAE CSVs"
        )
        total_copied += csv_count

# ===========================================================================================
# 2. HYPERPARAMETER SENSITIVITY (12 configs)
# ===========================================================================================

log("\n" + "=" * 80)
log("SECTION 2: HYPERPARAMETER SENSITIVITY")
log("=" * 80)

hyperparam_src = os.path.join(GDRIVE_BASE, "NeuroScan_Research", "Research_HyperparamSensitivity")
hyperparam_dst = os.path.join(BACKUP_ROOT, "hyperparameter_sensitivity")

csv_count = find_and_copy_csvs(hyperparam_src, hyperparam_dst, "Hyperparameter Sensitivity (12 configs)")
total_copied += csv_count

# ===========================================================================================
# 3. ARCHITECTURE ABLATIONS (from NeuroScan_PEDiMS_AblationStudies)
# ===========================================================================================

log("\n" + "=" * 80)
log("SECTION 3: ARCHITECTURE ABLATIONS")
log("=" * 80)

arch_ablation_src = os.path.join(GDRIVE_BASE, "NeuroScan_PEDiMS_AblationStudies")
arch_ablation_dst = os.path.join(BACKUP_ROOT, "architecture_ablations")

csv_count = find_and_copy_csvs(arch_ablation_src, arch_ablation_dst, "Architecture Ablations")
total_copied += csv_count

# ===========================================================================================
# 4. FINAL MODEL RESULTS
# ===========================================================================================

log("\n" + "=" * 80)
log("SECTION 4: FINAL MODEL RESULTS")
log("=" * 80)

final_model_src = os.path.join(GDRIVE_BASE, "NeuroScan_FinalModel_2.5D_MAE")
final_model_dst = os.path.join(BACKUP_ROOT, "final_model_results")

csv_count = find_and_copy_csvs(final_model_src, final_model_dst, "Final Model Results")
total_copied += csv_count

# ===========================================================================================
# 5. RESEARCH EXPERIMENTS
# ===========================================================================================

log("\n" + "=" * 80)
log("SECTION 5: RESEARCH EXPERIMENTS")
log("=" * 80)

research_src = os.path.join(GDRIVE_BASE, "NeuroScan_Research")
research_dst = os.path.join(BACKUP_ROOT, "research_experiments")

csv_count = find_and_copy_csvs(research_src, research_dst, "Research Experiments")
total_copied += csv_count

# ===========================================================================================
# 6. OTHER NEUROSCAN FOLDERS
# ===========================================================================================

log("\n" + "=" * 80)
log("SECTION 6: OTHER NEUROSCAN RESULTS")
log("=" * 80)

other_folders = [
    "NeuroScan_PEDiMS",
    "NeuroScan_PEDiMS_v2",
    "NeuroScan_PEDiMS_v3",
    "NeuroScan_2p5D",
    "NeuroScan"
]

for folder in other_folders:
    src = os.path.join(GDRIVE_BASE, folder)
    dst = os.path.join(BACKUP_ROOT, "legacy_experiments", folder)
    csv_count = find_and_copy_csvs(src, dst, f"{folder} CSVs")
    total_copied += csv_count

# ===========================================================================================
# 7. COPY SPECIFIC IMPORTANT FILES
# ===========================================================================================

log("\n" + "=" * 80)
log("SECTION 7: IMPORTANT CONFIGURATION FILES")
log("=" * 80)

# Copy any important config/results files
important_files = []

# Find all JSON result files
for root, dirs, files in os.walk(GDRIVE_BASE):
    for file in files:
        if file in ['results.json', 'config.json', 'experiment_config.json']:
            src_path = os.path.join(root, file)
            rel_path = os.path.relpath(src_path, GDRIVE_BASE)
            dst_path = os.path.join(BACKUP_ROOT, "configs", rel_path)
            
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            try:
                shutil.copy2(src_path, dst_path)
                log(f"   ✅ {rel_path}")
                total_copied += 1
            except Exception as e:
                log(f"   ❌ ERROR: {rel_path} - {e}")

# ===========================================================================================
# SUMMARY
# ===========================================================================================

log("\n" + "=" * 80)
log("MIGRATION SUMMARY")
log("=" * 80)

# Calculate backup size
backup_size_mb = get_folder_size(BACKUP_ROOT)
total_size_mb += backup_size_mb

log(f"\n📊 Total files copied: {total_copied}")
log(f"💾 Total size: {backup_size_mb:.1f} MB ({backup_size_mb/1024:.2f} GB)")
log(f"📂 Backup location: {BACKUP_ROOT}")

# Create summary file
summary_file = os.path.join(BACKUP_ROOT, "MIGRATION_SUMMARY.txt")
with open(summary_file, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("GOOGLE DRIVE MIGRATION SUMMARY\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Migration Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Source: {GDRIVE_BASE}\n")
    f.write(f"Destination: {BACKUP_ROOT}\n")
    f.write(f"Total Files: {total_copied}\n")
    f.write(f"Total Size: {backup_size_mb:.1f} MB ({backup_size_mb/1024:.2f} GB)\n\n")
    f.write("=" * 80 + "\n")
    f.write("FOLDER STRUCTURE:\n")
    f.write("=" * 80 + "\n")
    f.write("ablation_experiments/       - All 7 USALD ablation configs\n")
    f.write("hyperparameter_sensitivity/ - 12 k_slices × window_size configs\n")
    f.write("architecture_ablations/     - 6 architecture component ablations\n")
    f.write("final_model_results/        - Final model training results\n")
    f.write("research_experiments/       - All research experiments\n")
    f.write("legacy_experiments/         - Older NeuroScan experiments\n")
    f.write("configs/                    - JSON config files\n")
    f.write("=" * 80 + "\n")

log(f"\n✅ Summary saved to: {summary_file}")

# Create list of Google Drive folders that can be deleted
cleanup_file = os.path.join(EDI_BASE, "GOOGLE_DRIVE_CLEANUP_LIST.txt")
with open(cleanup_file, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("GOOGLE DRIVE FOLDERS - SAFE TO DELETE AFTER VERIFICATION\n")
    f.write("=" * 80 + "\n\n")
    f.write("⚠️ IMPORTANT: Verify backup completeness before deleting!\n\n")
    f.write("After verifying the backup at:\n")
    f.write(f"  {BACKUP_ROOT}\n\n")
    f.write("You can safely delete these folders from Google Drive:\n\n")
    
    all_folders = ablation_configs + [
        "NeuroScan_Research",
        "NeuroScan_PEDiMS_AblationStudies",
        "NeuroScan_FinalModel_2.5D_MAE",
    ] + other_folders
    
    for folder in all_folders:
        folder_path = os.path.join(GDRIVE_BASE, folder)
        if os.path.exists(folder_path):
            size_mb = get_folder_size(folder_path)
            f.write(f"  [ ] {folder} ({size_mb:.1f} MB / {size_mb/1024:.2f} GB)\n")
    
    f.write("\n" + "=" * 80 + "\n")
    f.write("ESTIMATED SPACE TO BE FREED: Check folder sizes above\n")
    f.write("=" * 80 + "\n")

log(f"\n✅ Cleanup list saved to: {cleanup_file}")

log("\n" + "=" * 80)
log("MIGRATION COMPLETE!")
log("=" * 80)
log("\n🎯 NEXT STEPS:")
log("1. Verify backup completeness in:")
log(f"   {BACKUP_ROOT}")
log("2. Check the cleanup list:")
log(f"   {cleanup_file}")
log("3. After verification, delete Google Drive folders to free space")
log("4. Keep Dataset folder on Google Drive (shared resource)")
log("\n" + "=" * 80)
