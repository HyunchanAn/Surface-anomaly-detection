from anomalib.engine import Engine
from anomalib.models import Patchcore
from pathlib import Path

def export():
    # Find the latest ckpt
    results_dir = Path("results/Patchcore/surface")
    if not results_dir.exists():
        print("Results directory not found.")
        return

    versions = sorted([d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("v")], key=lambda x: int(x.name[1:]), reverse=True)
    if not versions:
        print("No model found.")
        return
    
    ckpt_path = list(versions[0].glob("weights/lightning/*.ckpt"))[0]
    print(f"[INFO] Found checkpoint: {ckpt_path}")
    print("[INFO] Starting export to TorchScript (.pt)...")
    
    model = Patchcore(
        backbone="wide_resnet50_2",
        pre_trained=True,
        coreset_sampling_ratio=0.1,
        num_neighbors=9
    )
    
    engine = Engine(logger=False)
    
    # Export
    # This will create exported_models/Patchcore/model.pt
    exported_path = engine.export(
        model=model,
        export_type="torch",
        ckpt_path=str(ckpt_path),
        export_root="exported_models",
        input_size=(224, 224) 
    )
    
    print(f"[SUCCESS] Exported to {exported_path}")

if __name__ == "__main__":
    export()
