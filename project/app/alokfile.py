import os
import numpy as np

DATA_DIR = r"F:\projects\KSL\KSL_Backend\project\app\ksl_video_data_father" 
MIN_FRAMES = 10                    # threshold warning

print("\n📂 Inspecting folder:", DATA_DIR)
print("=" * 50)

total_files = 0
frame_counts = []

for file in sorted(os.listdir(DATA_DIR)):
    if not file.endswith(".npy"):
        continue

    path = os.path.join(DATA_DIR, file)
    data = np.load(path)

    total_files += 1
    frames, features = data.shape
    frame_counts.append(frames)

    print(f"📄 {file}")
    print(f"   Shape      : {data.shape}")
    print(f"   Frames     : {frames}")
    print(f"   Features   : {features}")

    if frames < MIN_FRAMES:
        print("   ⚠ WARNING  : Too few frames")

    print(f"   First frame (first 6 values): {data[0][:6]}")
    print("-" * 50)

# -------- Summary --------
print("\n📊 SUMMARY")
print("=" * 50)
print("Total files      :", total_files)
print("Min frames       :", min(frame_counts))
print("Max frames       :", max(frame_counts))
print("Average frames   :", sum(frame_counts) / len(frame_counts))
