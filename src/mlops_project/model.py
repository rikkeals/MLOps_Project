import json
import os
import subprocess
import sys
from pathlib import Path

import hydra
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


def update_config_from_plans_2d(project_root: Path, dataset_name: str, config_path: Path) -> None:
    """
    Reads nnUNetPlans.json for this dataset and writes the 2D configuration into cfg.nnunet.defaults.
    Keeps cfg.nnunet.override untouched so you can experiment without losing the baseline.
    """
    plans_path = (
        project_root
        / "data"
        / "nnUNet_preprocessed"
        / dataset_name
        / "nnUNetPlans.json"
    )

    if not plans_path.exists():
        raise FileNotFoundError(f"Could not find plans at: {plans_path}")

    with open(plans_path, "r") as f:
        plans = json.load(f)

    conf_2d = plans["configurations"]["2d"]

    # Keep only the hyperparams you actually care about in your project config
    extracted = {
        "data_identifier": conf_2d.get("data_identifier"),
        "batch_size": conf_2d.get("batch_size"),
        "patch_size": conf_2d.get("patch_size"),
        "spacing": conf_2d.get("spacing"),
        "normalization_schemes": conf_2d.get("normalization_schemes"),
        "architecture": conf_2d.get("architecture"),
    }

    # Load + update yaml
    cfg = OmegaConf.load(config_path)

    # Ensure structure exists
    if "model" not in cfg:
        cfg.model = {}
    cfg.model.configuration = "2d"
    cfg.model.defaults = extracted

    # Write back
    OmegaConf.save(cfg, config_path)
    print(f"Updated config defaults from plans (2d) -> {config_path}")



@hydra.main(version_base=None, config_path="../../configs", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    # Hydra changes the working dir. This returns the dir you launched python from (repo root typically).
    project_root = Path(hydra.utils.get_original_cwd())

    # 1) Set env vars (must be set before calling nnUNet CLI)
    set_nnunet_env(project_root)

    # 2) Read dataset id and configuration from config
    dataset_id = str(cfg.dataset.dataset_id)
    configuration = str(cfg.preprocessing.configuration)

    # 3) Run nnUNet CLI
    cmd = ["nnUNetv2_plan_and_preprocess", "-d", dataset_id, "--verify_dataset_integrity", "-c", configuration]

    print("\nRunning:", " ".join(cmd))

    try:
        subprocess.run(cmd, check=True, env=os.environ.copy())
    except FileNotFoundError:
        print("\nERROR: nnUNetv2_plan_and_preprocess not found on PATH.")
        print("Make sure you installed nnunetv2 in the same environment you're running this script from.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: nnUNet command failed with exit code {e.returncode}")
        sys.exit(e.returncode)

    # 4) Update config.yaml from the generated Plans_2d
    config_path = project_root / "configs" / "config.yaml"
    dataset_name = str(cfg.dataset.name)

    update_config_from_plans_2d(
        project_root=project_root,
        dataset_name=dataset_name,
        config_path=config_path,
    )


if __name__ == "__main__":
    main()
