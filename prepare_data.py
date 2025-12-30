import os
import shutil
import random
from pathlib import Path
import urllib.request
import zipfile

# Configuration
DATASET_ROOT = Path("datasets/custom")
TRAIN_GOOD_DIR = DATASET_ROOT / "train" / "good"
TEST_GOOD_DIR = DATASET_ROOT / "test" / "good"
TEST_BAD_DIR = DATASET_ROOT / "test" / "bad"

def create_structure():
    """Creates the necessary folder structure."""
    for d in [TRAIN_GOOD_DIR, TEST_GOOD_DIR, TEST_BAD_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Created directory structure at {DATASET_ROOT}")

def organize_user_data(source_dir):
    """
    Distributes user images from a source source_dir into train/test splits.
    Assumption: source_dir contains only 'good' (normal) images.
    """
    source_path = Path(source_dir)
    images = list(source_path.glob("*.jpg")) + list(source_path.glob("*.png"))
    
    if not images:
        print("[WARN] No images found in source directory.")
        return

    random.shuffle(images)
    
    # 80/20 Split
    split_idx = int(len(images) * 0.8)
    train_imgs = images[:split_idx]
    test_imgs = images[split_idx:]
    
    print(f"[INFO] Found {len(images)} images. Moving to dataset folders...")
    
    for img in train_imgs:
        shutil.copy(img, TRAIN_GOOD_DIR / img.name)
        
    for img in test_imgs:
        shutil.copy(img, TEST_GOOD_DIR / img.name)
        
    print("[SUCCESS] Data organized. Now add some 'bad' images to 'datasets/custom/test/bad' for validation.")

def download_sample_data():
    """
    Downloads a sample dataset (e.g. Hazelnut from MVTec or similar) if user has no data.
    Actually, let's just create creating dummy files or instructions since direct downloading large datasets might be fragile.
    """
    print("[INFO] No data source provided. Please put your normal images in 'datasets/raw_images' and run this script again.")
    print("      Or manually copy them to 'datasets/custom/train/good'.")

if __name__ == "__main__":
    create_structure()
    
    # Check if user has put raw images somewhere to organize
    raw_dir = Path("datasets/raw_images")
    if raw_dir.exists():
        organize_user_data(raw_dir)
    else:
        # Create raw dir for user convenience
        raw_dir.mkdir(exist_ok=True)
        print(f"[TIP] Put your 50+ normal surface photos in '{raw_dir}' and run 'python prepare_data.py' to auto-split them.")
