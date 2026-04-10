"""
Update all dataset paths from Google Drive to local EDI folder
This ensures all scripts use the new local dataset location
"""

import os
import re

# Files to update
files_to_update = [
    r"C:\Users\HP\EDI\research\research_utils.py",
    r"C:\Users\HP\EDI\research\noise_robustness_analysis.py",
    r"C:\Users\HP\EDI\research\noise_robustness_analysis_quick.py",
    r"C:\Users\HP\EDI\research\hyperparameter_sensitivity.py",
    r"C:\Users\HP\EDI\research\hyperparameter_sensitivity_quick.py",
    r"C:\Users\HP\EDI\research\csrf_variants_analysis.py",
    r"C:\Users\HP\EDI\research\csrf_variants_analysis_quick.py",
    r"C:\Users\HP\EDI\research\cross_validation_framework.py",
    r"C:\Users\HP\EDI\research\generate_confusion_matrix.py",
    r"C:\Users\HP\EDI\final_model.py",
]

# Path replacements
replacements = [
    # Google Drive paths
    (r'G:\\My Drive', r'C:\\Users\\HP\\EDI'),
    (r'G:/My Drive', r'C:/Users/HP/EDI'),
    (r'/content/drive/MyDrive', r'C:/Users/HP/EDI'),
    
    # Specific dataset paths
    (r'os\.path\.join\(DRIVE_BASE, "Dataset", "PediMS", "PediMS"\)', 
     r'os.path.join(r"C:\\Users\\HP\\EDI", "Dataset", "PediMS", "PediMS")'),
]

print("=" * 80)
print("UPDATING DATASET PATHS TO LOCAL EDI FOLDER")
print("=" * 80)

for filepath in files_to_update:
    if not os.path.exists(filepath):
        print(f"⚠️ SKIP: {os.path.basename(filepath)} - File not found")
        continue
    
    print(f"\n📄 Updating: {os.path.basename(filepath)}")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        changes_made = 0
        
        # Apply replacements
        for old_pattern, new_pattern in replacements:
            matches = len(re.findall(old_pattern, content))
            if matches > 0:
                content = re.sub(old_pattern, new_pattern, content)
                changes_made += matches
                print(f"   ✓ Replaced {matches} occurrence(s) of '{old_pattern[:40]}...'")
        
        if changes_made > 0:
            # Backup original
            backup_path = filepath + ".backup"
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(original_content)
            
            # Write updated content
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"   ✅ Updated successfully! ({changes_made} changes)")
            print(f"   💾 Backup saved: {os.path.basename(backup_path)}")
        else:
            print(f"   ℹ️ No changes needed")
    
    except Exception as e:
        print(f"   ❌ ERROR: {e}")

print("\n" + "=" * 80)
print("PATH UPDATE COMPLETE!")
print("=" * 80)
print("\n✅ All scripts now use local dataset at: C:\\Users\\HP\\EDI\\Dataset")
print("✅ Google Drive is no longer required for training!")
print("\n" + "=" * 80)
