"""
Migrate Dataset folder from Google Drive to local EDI folder
This will free up significant Google Drive space
"""

import os
import shutil
from pathlib import Path
from datetime import datetime

# Paths
GDRIVE_DATASET = r"G:\My Drive\Dataset"
LOCAL_DATASET = r"C:\Users\HP\EDI\Dataset"
LOG_FILE = r"C:\Users\HP\EDI\dataset_migration_log.txt"

def log(message):
    """Log to both console and file"""
    print(message)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")

def get_folder_size(folder_path):
    """Calculate folder size in GB"""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(folder_path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
    except Exception as e:
        log(f"   ⚠️ Error calculating size: {e}")
    return total_size / (1024 * 1024 * 1024)  # Convert to GB

def copy_with_progress(src, dst):
    """Copy folder with progress tracking"""
    log("\n" + "=" * 80)
    log("DATASET MIGRATION - START")
    log("=" * 80)
    
    if not os.path.exists(src):
        log(f"❌ ERROR: Source dataset not found: {src}")
        return False
    
    # Calculate size
    log(f"\n📊 Calculating dataset size...")
    size_gb = get_folder_size(src)
    log(f"Dataset size: {size_gb:.2f} GB")
    
    # Check if destination exists
    if os.path.exists(dst):
        log(f"\n⚠️ WARNING: Destination already exists: {dst}")
        log("Backing up existing dataset...")
        backup_path = dst + "_old_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.move(dst, backup_path)
        log(f"Old dataset moved to: {backup_path}")
    
    # Copy dataset
    log(f"\n📂 Copying dataset...")
    log(f"   From: {src}")
    log(f"   To: {dst}")
    log(f"\n⏳ This will take several minutes for {size_gb:.2f} GB...")
    log("Please wait...\n")
    
    try:
        start_time = datetime.now()
        
        # Copy with progress
        total_files = sum([len(files) for _, _, files in os.walk(src)])
        copied_files = 0
        
        def copy_tree_with_progress(src_dir, dst_dir):
            nonlocal copied_files
            os.makedirs(dst_dir, exist_ok=True)
            
            for item in os.listdir(src_dir):
                src_item = os.path.join(src_dir, item)
                dst_item = os.path.join(dst_dir, item)
                
                if os.path.isdir(src_item):
                    copy_tree_with_progress(src_item, dst_item)
                else:
                    shutil.copy2(src_item, dst_item)
                    copied_files += 1
                    if copied_files % 100 == 0:  # Progress every 100 files
                        progress = (copied_files / total_files) * 100
                        log(f"Progress: {copied_files}/{total_files} files ({progress:.1f}%)")
        
        copy_tree_with_progress(src, dst)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        log(f"\n✅ Dataset copied successfully!")
        log(f"Total files copied: {copied_files}")
        log(f"Time taken: {elapsed/60:.1f} minutes")
        
        # Verify copy
        log(f"\n🔍 Verifying copy...")
        dst_size = get_folder_size(dst)
        if abs(size_gb - dst_size) < 0.01:  # Allow 10MB difference
            log(f"✅ Verification successful! Sizes match ({dst_size:.2f} GB)")
            return True
        else:
            log(f"⚠️ WARNING: Size mismatch! Source: {size_gb:.2f} GB, Dest: {dst_size:.2f} GB")
            log(f"Please verify manually before deleting from Google Drive!")
            return False
            
    except Exception as e:
        log(f"\n❌ ERROR during copy: {e}")
        return False

# ===========================================================================================
# MAIN EXECUTION
# ===========================================================================================

log("\n" + "=" * 80)
log("DATASET MIGRATION SCRIPT")
log("=" * 80)
log(f"This will copy the Dataset folder from Google Drive to local storage.")
log(f"Estimated time: 5-15 minutes depending on dataset size.")
log("=" * 80)

# Auto-proceed (no user input needed when run from script)
log("\n🚀 Starting automatic migration...")

success = copy_with_progress(GDRIVE_DATASET, LOCAL_DATASET)

if success:
    log("\n" + "=" * 80)
    log("MIGRATION COMPLETE!")
    log("=" * 80)
    log(f"\n✅ Dataset is now available at: {LOCAL_DATASET}")
    log(f"\n🎯 NEXT STEPS:")
    log(f"1. Verify the dataset by checking a few files in: {LOCAL_DATASET}")
    log(f"2. Update final_model.py to use local dataset path")
    log(f"3. Test training with local dataset")
    log(f"4. After verification, delete from Google Drive to free space")
    log(f"\n⚠️ IMPORTANT: Update DATA_PATH in final_model.py:")
    log(f'   Old: DATA_PATH = os.path.join(DRIVE_BASE, "Dataset", "PediMS", "PediMS")')
    log(f'   New: DATA_PATH = r"C:\\Users\\HP\\EDI\\Dataset\\PediMS\\PediMS"')
    log("\n" + "=" * 80)
else:
    log("\n" + "=" * 80)
    log("MIGRATION FAILED OR INCOMPLETE")
    log("=" * 80)
    log("\nPlease check the errors above and try again.")
    log("Do NOT delete from Google Drive until copy is verified!")
    log("=" * 80)
