# Surface Anomaly Detection System

## Project Overview
This project implements a surface defect detection system using Deep Learning (Anomalib PatchCore).
It is designed to run on a machine with a GPU (e.g., RTX 4060).

## 🚀 Quick Start (On your Home Machine)

### 1. Environment Setup
Refer to `REQUIREMENTS.md` for detailed package installation.
```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Data Preparation
You need about 50 "Good" (Normal) images of your surface (film, fabric, coating, etc.).

**Option A: Use Your Own Data**
1. Create a folder `datasets/raw_images`.
2. Put your 50 normal images inside it.
3. Run the helper script:
   ```bash
   python prepare_data.py
   ```

**Option B: Auto-Download Sample Data (KolektorSDD)**
If you don't have images yet, run:
```bash
python prepare_data.py --download
```
*This will download a sample surface dataset and organize it automatically.*

*This will automatically split them into `train/good` and `test/good`.*

4. (Optional) If you have defect images, put them manually into `datasets/custom/test/bad`.

### 3. Training
Run the training script. It will load `configs/surface_config.yaml`.
```bash
python train.py
```
*On RTX 4060, this should take less than 1-2 minutes.*
Artifacts/Model will be saved in `results/`.

### 4. Run Demo UI
Start the web interface to visualize results.
```bash
streamlit run app.py
```
Open your browser to `http://localhost:8501`.

## Directory Structure
- `datasets/`: Data storage.
- `configs/`: Model configuration (Hyperparameters).
- `models/`: Where trained `.ckpt` files are saved.
- `train.py`: Main training entry point.
- `app.py`: Streamlit Web App.
