from pathlib import Path
import os
import subprocess
import sys
import hydra
from omegaconf import DictConfig


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


@hydra.main(version_base=None, config_path="../../configs", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    # Hydra changes the working dir. This returns the dir you launched python from (repo root typically).
    project_root = Path(hydra.utils.get_original_cwd())

    # 1) Set env vars (must be set before calling nnUNet CLI)
    set_nnunet_env(project_root)

    # 2) Read dataset id from config
    dataset_id = str(cfg.dataset.dataset_id)

    # 3) Run nnUNet CLI
    cmd = ["nnUNetv2_plan_and_preprocess", "-d", dataset_id, "--verify_dataset_integrity", "-c", "2d"]
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


if __name__ == "__main__":
    main()
