#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import os
import shutil
import subprocess
import sys


# -------------------------
# Defaults
# -------------------------
DATASET_NAME = "Dataset621_Hippocampus"
PRED_FOLDER = Path("outputs/inference") / DATASET_NAME
OUTPUT_METRICS = Path("outputs/metrics.json")
NUM_PROCESSES = 4


def die(msg: str) -> None:
    print(f"[ERROR] {msg}")
    sys.exit(1)


def list_nii(folder: Path) -> set[str]:
    return {p.name for p in folder.glob("*.nii.gz")}


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    # Make nnU-Net paths work when teacher runs "python src/mlops_project/evaluate.py"
    os.environ.setdefault("nnUNet_raw", str(repo_root / "data/nnUNet_raw"))
    os.environ.setdefault("nnUNet_preprocessed", str(repo_root / "data/nnUNet_preprocessed"))
    os.environ.setdefault("nnUNet_results", str(repo_root / "data/nnUNet_results"))

    nnunet_raw_ds = Path(os.environ["nnUNet_raw"]) / DATASET_NAME
    nnunet_preproc_ds = Path(os.environ["nnUNet_preprocessed"]) / DATASET_NAME

    # Pick ground-truth folder
    gt_ts = nnunet_raw_ds / "labelsTs"
    gt_tr = nnunet_raw_ds / "labelsTr"
    if gt_ts.exists():
        gt_folder = gt_ts
    elif gt_tr.exists():
        gt_folder = gt_tr
    else:
        die(
            "No ground-truth labels found. Expected one of:\n"
            f"  {gt_ts}\n  {gt_tr}\n"
            "Make sure your nnUNet_raw folder contains labels."
        )

    # Find dataset.json + plans
    dataset_json = nnunet_preproc_ds / "dataset.json"
    if not dataset_json.exists():
        die(f"dataset.json not found at: {dataset_json}")

    plans_candidates = [
        nnunet_preproc_ds / "plans.json",
        nnunet_preproc_ds / "nnUNetPlans.json",
    ]
    plans_json = next((p for p in plans_candidates if p.exists()), None)
    if plans_json is None:
        die(
            "Could not find plans file. Tried:\n  "
            + "\n  ".join(str(p) for p in plans_candidates)
        )

    # Check predictions
    if not PRED_FOLDER.exists():
        die(f"Prediction folder not found: {PRED_FOLDER}")

    gt_files = list_nii(gt_folder)
    pred_files = list_nii(PRED_FOLDER)
    if not gt_files:
        die(f"No .nii.gz files found in GT folder: {gt_folder}")
    if not pred_files:
        die(f"No .nii.gz files found in prediction folder: {PRED_FOLDER}")

    overlap = gt_files.intersection(pred_files)
    if not overlap:
        die(
            "No matching filenames between GT and predictions.\n"
            f"GT folder:   {gt_folder}\n"
            f"Pred folder: {PRED_FOLDER}\n\n"
            f"Example GT:   {sorted(gt_files)[:3]}\n"
            f"Example Pred: {sorted(pred_files)[:3]}\n\n"
            "Fix: run inference on the same split you evaluate on."
        )

    exe = shutil.which("nnUNetv2_evaluate_folder")
    if exe is None:
        die("nnUNetv2_evaluate_folder not found in PATH (is nnunetv2 installed in this env?).")

    OUTPUT_METRICS.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        exe,
        str(gt_folder.resolve()),
        str(PRED_FOLDER.resolve()),
        "-djfile",
        str(dataset_json.resolve()),
        "-pfile",
        str(plans_json.resolve()),
        "-o",
        str(OUTPUT_METRICS.resolve()),
        "-np",
        str(NUM_PROCESSES),
    ]

    print("Running nnU-Net evaluation:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

    print("Done.")
    print(f"Metrics written to: {OUTPUT_METRICS}")


if __name__ == "__main__":
    main()
