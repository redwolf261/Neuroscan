import os

# Test Google Drive detection
possible_drives = [chr(i) + ":" for i in range(65, 91)]
DRIVE_BASE = None

for drive in possible_drives:
    if os.path.exists(drive):
        drive_path = os.path.join(drive, os.sep, "My Drive")
        if os.path.exists(drive_path):
            DRIVE_BASE = drive_path
            print(f"✓ Found Google Drive: {DRIVE_BASE}")
            break

if DRIVE_BASE is None:
    print("✗ Google Drive not found, using fallback")
    DRIVE_BASE = os.path.join(os.path.expanduser("~"), "MyDrive_Local")

DATA_PATH = os.path.join(DRIVE_BASE, "Dataset", "PediMS", "PediMS")
print(f"\nDATA_PATH: {DATA_PATH}")
print(f"Data exists: {os.path.exists(DATA_PATH)}")

if os.path.exists(DATA_PATH):
    subjects = sorted([f for f in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, f))])
    print(f"Number of subjects: {len(subjects)}")
    print(f"First 5 subjects: {subjects[:5]}")
else:
    print("⚠️  DATA_PATH does not exist!")
