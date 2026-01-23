import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import hydra
import nibabel as nib
import numpy as np
from omegaconf import DictConfig, OmegaConf


def set_nnunet_env(project_root: Path) -> None:
    """Force nnUNet env vars to be relative to the project root."""
    data_dir = project_root / "data"
    os.environ["nnUNet_raw"] = str(data_dir / "nnUNet_raw")
    os.environ["nnUNet_preprocessed"] = str(data_dir / "nnUNet_preprocessed")
    os.environ["nnUNet_results"] = str(data_dir / "nnUNet_results")

    print("nnU-Net environment variables set:")
    print("  nnUNet_raw:", os.environ["nnUNet_raw"])
    print("  nnUNet_preprocessed:", os.environ["nnUNet_preprocessed"])
    print("  nnUNet_results:", os.environ["nnUNet_results"])

def dice_binary(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-8) -> float:
    """Dice for boolean masks."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    return (2.0 * inter + eps) / (pred.sum() + gt.sum() + eps)


def dice_per_class(seg_pred: np.ndarray, seg_gt: np.ndarray, classes: List[int]) -> Dict[int, float]:
    """Dice for each integer label in `classes`."""
    out = {}
    for c in classes:
        out[c] = dice_binary(seg_pred == c, seg_gt == c)
    return out


def evaluate_dice(
        pred_dir: Path,
        gt_dir: Path,
        classes: List[int]
        ) -> Tuple[Dict[int, float], Dict[str, Dict[int, float]]]:
    """
    Evaluate Dice for each file in pred_dir against gt_dir.
    Returns:
      - mean dice per class
      - per-case dice per class (keyed by case id / filename stem)
    """
    pred_files = sorted(list(pred_dir.glob("*.nii.gz")))
    if not pred_files:
        raise FileNotFoundError(f"No .nii.gz predictions found in: {pred_dir}")

    per_case: Dict[str, Dict[int, float]] = {}

    for pf in pred_files:
        # nnU-Net usually keeps same filename for prediction and GT
        gt_path = gt_dir / pf.name
        if not gt_path.exists():
            # fallback: sometimes GT uses .nii instead of .nii.gz or naming differs
            # you can extend this logic if needed
            raise FileNotFoundError(f"Missing ground truth for prediction: {pf.name} (expected {gt_path})")

        pred = nib.load(str(pf)).get_fdata().astype(np.int32)
        gt = nib.load(str(gt_path)).get_fdata().astype(np.int32)

        if pred.shape != gt.shape:
            raise ValueError(f"Shape mismatch for {pf.name}: pred {pred.shape} vs gt {gt.shape}")

        case_id = pf.name.replace(".nii.gz", "")
        per_case[case_id] = dice_per_class(pred, gt, classes)

    # mean per class
    mean_per_class: Dict[int, float] = {}
    for c in classes:
        mean_per_class[c] = float(np.mean([per_case[k][c] for k in per_case]))

    return mean_per_class, per_case

@hydra.main(version_base=None, config_path="../../configs", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    # Hydra changes the working dir. This returns the dir you launched python from (repo root typically).
    project_root = Path(hydra.utils.get_original_cwd())

    input_folder = project_root / "data/nnUNet_raw/Dataset621_Hippocampus/ImagesTr"
    output_folder = project_root / "data/nnUNet_predictions/Dataset621_Hippocampus/predictionsTr"
    labels_folder = project_root / "data/nnUNet_raw/Dataset621_Hippocampus/labelsTr"

    # 1) Set env vars (must be set before calling nnUNet CLI)
    set_nnunet_env(project_root)

    # 2) Read dataset input folder, output folder, id and configuration from config
    dataset_id = str(cfg.dataset.dataset_id)
    configuration = str(cfg.model.configuration)

    # 3) Run nnUNet CLI
    trainer = "nnUNetTrainer_5epochs"
    plans = "nnUNetPlans"
    fold = "0"

    cmd = [
        "nnUNetv2_predict",
        "-i", str(input_folder),
        "-o", str(output_folder),
        "-d", str(dataset_id),
        "-c", str(configuration),
        "-f", fold,
        "-tr", trainer,
        "-p", plans,
    ]


    print("\nPrediction started:", " ".join(cmd))

    try:
        subprocess.run(cmd, check=True, env=os.environ.copy())
    except FileNotFoundError:
        print("\nERROR: nnUNetv2_plan_and_preprocess not found on PATH.")
        print("Make sure you installed nnunetv2 in the same environment you're running this script from.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: nnUNet command failed with exit code {e.returncode}")
        sys.exit(e.returncode)

        # 4) Dice evaluation on predictions vs labelsTr
    print("\nComputing Dice scores vs labelsTr...")

    # nnU-Net Hippocampus usually has labels: 0=background, 1=anterior, 2=posterior (or left/right depending on dataset)
    # Adjust if your dataset uses different label IDs.
    classes = [1, 2]

    mean_dice, per_case = evaluate_dice(output_folder, labels_folder, classes)

    print("\n=== Dice results (mean over cases) ===")
    for c in classes:
        print(f"Class {c}: {mean_dice[c]:.4f}")

    overall_mean = float(np.mean([mean_dice[c] for c in classes]))
    print(f"Mean over classes {classes}: {overall_mean:.4f}")

    # Optional: save per-case scores
    results_path = project_root / "reports" / "dice_scores.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({"mean_per_class": mean_dice, "per_case": per_case}, f, indent=2)
    print(f"\nSaved Dice scores to: {results_path}")



if __name__ == "__main__":
    main()
