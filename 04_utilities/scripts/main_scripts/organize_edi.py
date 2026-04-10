"""
ORGANIZE EDI FOLDER - Complete Reorganization
==============================================
Moves files into logical folders for clean organization

Structure:
    scripts/          - All Python runner scripts
    docs/             - All markdown documentation
    backups/          - All backup files
    logs/             - All log and text files
    models/           - Model architecture and training scripts
    tests/            - Test and verification scripts
    
Existing folders preserved:
    research/         - Research experiments
    ablation/         - Ablation checkpoints
    ablation_results/ - Ablation results
    Dataset/          - Training dataset
    csv_data/         - CSV data files
    config/           - Configuration files
    TODO/             - Task tracking
"""

import os
import shutil
import fnmatch
from pathlib import Path

EDI_ROOT = r"C:\Users\HP\EDI"

# Define organization structure
ORGANIZATION = {
    "scripts": {
        "description": "Python runner and automation scripts",
        "patterns": [
            "run_*.py",
            "collect_*.py",
            "migrate_*.py",
            "update_*.py",
            "check_*.py",
            "generate_*.py",
            "organize_*.py",
        ]
    },
    "docs": {
        "description": "Markdown documentation and guides",
        "patterns": [
            "*.md",
        ],
        "exclude": ["README.md"]  # Keep README in root
    },
    "backups": {
        "description": "Backup files",
        "patterns": [
            "*.backup",
        ]
    },
    "logs": {
        "description": "Log files and execution records",
        "patterns": [
            "*.log",
            "*_log.txt",
            "dataset_copy_log.txt",
            "dataset_migration_log.txt",
            "migration_log.txt",
        ]
    },
    "models": {
        "description": "Model architecture and training scripts",
        "patterns": [
            "final_model.py",
            "inference.py",
        ]
    },
    "tests": {
        "description": "Test and verification scripts",
        "patterns": [
            "test_*.py",
            "verify_*.py",
            "quick_measure.py",
        ]
    },
    "utilities": {
        "description": "Utility scripts and tools",
        "patterns": [
            "trial.py",
            "trial_local.py",
        ]
    },
}

def organize_files():
    """Organize files into folders"""
    
    print("=" * 80)
    print("EDI FOLDER ORGANIZATION")
    print("=" * 80)
    print("\nCreating organized folder structure...")
    
    # Create folders
    for folder, config in ORGANIZATION.items():
        folder_path = os.path.join(EDI_ROOT, folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"  ✅ Created: {folder}/ - {config['description']}")
    
    # Move files
    print("\n" + "=" * 80)
    print("MOVING FILES")
    print("=" * 80)
    
    moved_count = 0
    skipped_count = 0
    
    for folder, config in ORGANIZATION.items():
        folder_path = os.path.join(EDI_ROOT, folder)
        exclude_list = config.get('exclude', [])
        
        print(f"\n📁 {folder}/")
        
        for pattern in config['patterns']:
            # Find matching files in root
            for item in os.listdir(EDI_ROOT):
                item_path = os.path.join(EDI_ROOT, item)
                
                # Skip if not a file
                if not os.path.isfile(item_path):
                    continue
                
                # Skip if in exclude list
                if item in exclude_list:
                    continue
                
                # Check if matches pattern
                if fnmatch.fnmatch(item, pattern):
                    dest_path = os.path.join(folder_path, item)
                    
                    # Don't overwrite if already exists
                    if os.path.exists(dest_path):
                        print(f"  ⚠️ {item} (already in {folder}/)")
                        skipped_count += 1
                        continue
                    
                    try:
                        shutil.move(item_path, dest_path)
                        print(f"  ✅ {item}")
                        moved_count += 1
                    except Exception as e:
                        print(f"  ❌ ERROR: {item} - {e}")
    
    # Summary
    print("\n" + "=" * 80)
    print("ORGANIZATION COMPLETE")
    print("=" * 80)
    print(f"\n📊 Statistics:")
    print(f"  - Files moved: {moved_count}")
    print(f"  - Files skipped: {skipped_count}")
    
    # Show final structure
    print("\n📁 FINAL FOLDER STRUCTURE:")
    print("=" * 80)
    
    all_folders = []
    
    # New organized folders
    for folder in sorted(ORGANIZATION.keys()):
        folder_path = os.path.join(EDI_ROOT, folder)
        if os.path.exists(folder_path):
            files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
            all_folders.append((folder, len(files), ORGANIZATION[folder]['description']))
    
    # Existing special folders
    special_folders = {
        "research": "Research experiments",
        "ablation": "Ablation checkpoints",
        "ablation_results": "Ablation results",
        "Dataset": "Training dataset (PediMS)",
        "csv_data": "CSV data files",
        "config": "Configuration files",
        "TODO": "Task tracking",
        "google_drive_backup": "Google Drive backups",
    }
    
    for folder, desc in special_folders.items():
        folder_path = os.path.join(EDI_ROOT, folder)
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            try:
                files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
                all_folders.append((folder, len(files), desc))
            except:
                pass
    
    # Print sorted list
    for folder, count, desc in sorted(all_folders):
        print(f"  {folder:25s} ({count:3d} files) - {desc}")
    
    # Files remaining in root
    root_files = [f for f in os.listdir(EDI_ROOT) if os.path.isfile(os.path.join(EDI_ROOT, f))]
    if root_files:
        print(f"\n  ROOT ({len(root_files)} files) - Main entry files")
        for f in sorted(root_files)[:5]:
            print(f"    - {f}")
        if len(root_files) > 5:
            print(f"    ... and {len(root_files) - 5} more")
    
    print("\n" + "=" * 80)
    print("✅ EDI folder is now organized and clean!")
    print("=" * 80)

if __name__ == "__main__":
    organize_files()
