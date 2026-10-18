"""Restructure raw dataset into ImageFolder format (NORMAL vs PNEUMONIA)."""
import os
import shutil
import random
from pathlib import Path

# SRC ko "data" set karein kyunke ab raw files data/ me hi hain
SRC = Path("data")
DST = Path("data/chest_xray")

random.seed(42)

def main():
    if not (SRC / "train").exists() or not (SRC / "test").exists():
        print(f"Error: Raw dataset 'train' aur 'test' folders {SRC} mein nahi mile.")
        return

    if DST.exists():
        print(f"Cleaning existing {DST}...")
        shutil.rmtree(DST)

    print("Processing Training Data...")
    
    # Collect files
    bacterial_dir = SRC / "train" / "BACTERIAL"
    viral_dir = SRC / "train" / "VIRAL"
    normal_dir = SRC / "train" / "NORMAL"

    if not bacterial_dir.exists():
         print(f"Error: {bacterial_dir} nahi mila. Kya folder structure sahi hai?")
         return

    bacterial = [bacterial_dir / f for f in os.listdir(bacterial_dir) if not f.startswith('.')]
    viral = [viral_dir / f for f in os.listdir(viral_dir) if not f.startswith('.')]
    normal = [normal_dir / f for f in os.listdir(normal_dir) if not f.startswith('.')]

    pneumonia_files = bacterial + viral # Combine Bacteria + Virus
    normal_files = normal

    # Shuffle
    random.shuffle(pneumonia_files)
    random.shuffle(normal_files)

    # Split 90% Train, 10% Val
    split_idx_p = int(len(pneumonia_files) * 0.9)
    split_idx_n = int(len(normal_files) * 0.9)

    train_p, val_p = pneumonia_files[:split_idx_p], pneumonia_files[split_idx_p:]
    train_n, val_n = normal_files[:split_idx_n], normal_files[split_idx_n:]

    # Create directories and copy
    for split_name, files_p, files_n in [
        ("train", train_p, train_n),
        ("val", val_p, val_n)
    ]:
        (DST / split_name / "PNEUMONIA").mkdir(parents=True, exist_ok=True)
        (DST / split_name / "NORMAL").mkdir(parents=True, exist_ok=True)

        for f in files_p:
            shutil.copy(f, DST / split_name / "PNEUMONIA" / f.name)
        for f in files_n:
            shutil.copy(f, DST / split_name / "NORMAL" / f.name)

    print("Processing Test Data...")
    (DST / "test" / "PNEUMONIA").mkdir(parents=True, exist_ok=True)
    (DST / "test" / "NORMAL").mkdir(parents=True, exist_ok=True)

    test_dir = SRC / "test"
    test_files = os.listdir(test_dir)
    for f in test_files:
        if f.startswith('.'): continue
        f_lower = f.lower()
        # Filename ke basis par class check karein
        if "_normal_" in f_lower or "normal" in f_lower:
            dst_cls = "NORMAL"
        elif "_bacteria_" in f_lower or "_virus_" in f_lower:
            dst_cls = "PNEUMONIA"
        else:
            dst_cls = "NORMAL" # Fallback
            
        shutil.copy(test_dir / f, DST / "test" / dst_cls / f)

    print("\nDataset successfully restructured!")
    for split in ["train", "val", "test"]:
        for cls in ["NORMAL", "PNEUMONIA"]:
            count = len(list((DST / split / cls).glob("*.*")))
            print(f"{split}/{cls}: {count} images")

if __name__ == "__main__":
    main()