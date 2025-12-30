import os
from anomalib.engine import Engine
from anomalib.utils.configs import AnomalibConfig
from pathlib import Path

def train():
    print("[INFO] Starting Surface Anomaly Detection Training Pipeline...")
    
    # 1. Config Load
    config_path = Path("configs/surface_config.yaml")
    if not config_path.exists():
        print(f"[ERROR] Config file not found at {config_path}")
        return

    # Check data is ready
    data_path = Path("datasets/custom/train/good") 
    # NOTE: Anomalib folder dataset defaults to expecting 'train/good' structure.
    # We might need to help the user rearrange if they just dumped pics in 'datasets/custom/normal'.
    
    # Let's ensure output directory exists
    os.makedirs("results", exist_ok=True)

    print("[INFO] Loading Config and initializing Engine...")
    # Initialize Engine
    # Anomalib 1.0+ simplifies this. We can often just pass the config path to the CLI,
    # but strictly in python:
    engine = Engine(
        config_path=str(config_path),
        default_root_dir="results",
        task="classification",  # or segmentation
    )

    # 2. Train
    print("[INFO] Beginning Training (Fitting)...")
    try:
        # If dataset structure is standard, this works automatically
        engine.fit()
        print("[SUCCESS] Training complete. Model saved in 'results/'.")
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        print("[TIP] Ensure your data is organized as: datasets/custom/train/good/coord_img_01.jpg")

if __name__ == "__main__":
    train()
