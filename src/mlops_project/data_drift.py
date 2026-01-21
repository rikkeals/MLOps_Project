from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from evidently.legacy.metrics import DataDriftTable
from evidently.legacy.report import Report

# --- Dataset paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET = PROJECT_ROOT / "data/nnUNet_raw/Dataset621_Hippocampus"
IMAGES_TR = DATASET / "imagesTr"
IMAGES_TS = DATASET / "imagesTs"

# --- Helpers ---
def load_volume(path: Path) -> np.ndarray:
    img = nib.load(path)
    return img.get_fdata()

def extract_features(volume: np.ndarray) -> dict:
    gradients = np.gradient(volume)
    return {
        "brightness": float(np.mean(volume)),
        "contrast": float(np.std(volume)),
        "sharpness": float(np.mean([np.mean(np.abs(g)) for g in gradients])),
    }

def build_feature_df(folder: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(folder.glob("*.nii.gz")):
        vol = load_volume(f)
        feats = extract_features(vol)
        rows.append(feats)
    return pd.DataFrame(rows)

# --- Create feature tables ---
ref_df = build_feature_df(IMAGES_TR)
cur_df = build_feature_df(IMAGES_TS)

# --- Evidently: train vs test drift ---
report = Report(metrics=[DataDriftTable()])
report.run(
    reference_data=ref_df,
    current_data=cur_df,
)
report.save_html("reports/drift_train_vs_test.html")
