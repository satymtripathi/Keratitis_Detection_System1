# Keratitis Detection System

This repository contains a hybrid Multi-Instance Learning (MIL) and Tabular feature training and inference pipeline for Keratitis detection.

## Project Structure
- `Model1/`: Core training, batch inference, and Streamlit UI scripts.
- `Limbus_Crop_Segmentation_System/`: Pre-trained segmentation models (U-Net++) for ROI extraction.
- `training_results/`: Pre-trained classifier weights and feature metadata.
- `requirements.txt`: Python package dependencies.

## Setup
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
### 1. Interactive Dashboard (Streamlit)
To run the expert UI for single-image diagnosis:
```bash
streamlit run Model1/04_modelUI.py
```

### 2. Batch Inference
To run inference on a folder of images:
```bash
python Model1/03_Test_batch_single.py --seg_ckpt Limbus_Crop_Segmentation_System/model_limbus_crop_unetpp_weighted.pth --cls_ckpt training_results/checkpoints/best.pth --input path/to/images --out_dir output_results
```

## Data Flow
The system uses a 3-stage pipeline:
1. **Segmentation**: Extracts global eye crop and polar limbus slices.
2. **Feature Extraction**: Computes clinical texture features (LBP, FFT, Entropy) from the best quality slices.
3. **Hybrid Model**: Fuses deep features from the global eye, multiple slices (via Attention MIL), and tabular features for the final diagnosis.
