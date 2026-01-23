from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import hydra
from loguru import logger
from omegaconf import DictConfig


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def require_dir(path: Path, name: str) -> None:
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(
            f"{name} directory is missing:\n  {path}\n"
            "This directory must already exist. "
            "It will appear only if the corresponding nnU-Net step ran successfully."
        )


def run_and_tee(cmd: list[str], log_path: Path, env: dict[str, str]) -> None:
    with log_path.open("a", encoding="utf-8") as f:
        f.write("\n" + "=" * 120 + "\n")
        f.write(f"COMMAND: {' '.join(cmd)}\n")
        f.write("=" * 120 + "\n")

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,              # line-buffered
            universal_newlines=True,
            env={**env, "PYTHONUNBUFFERED": "1"},
        )

        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            f.write(line)
            f.flush()

        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"Command failed with exit code {rc}: {' '.join(cmd)}")


def maybe_init_wandb(cfg: DictConfig) -> Optional[object]:
    if not cfg.get("wandb") or not bool(cfg.wandb.get("enabled", False)):
        return None

    try:
        import wandb  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "wandb is enabled in config but not installed. Install with:\n"
            "  pip install wandb\n"
        ) from e

    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.wandb.run_name,
        job_type=cfg.wandb.job_type,
        config={
            "dataset_id": int(cfg.training.dataset_id),
            "nnunet_config": cfg.training.nnunet_config,
            "fold": cfg.training.fold,
            "trainer": cfg.training.trainer,
            "device": cfg.training.device,
        },
    )

    return run


def parse_epoch_metrics(log_path: Path) -> list[dict]:
    """
    Parse per-epoch metrics from nnU-Net log.
    Returns a list of dicts: [{epoch, train_loss, val_loss, pseudo_dice}]
    """
    epoch_data = []
    current = {}

    for line in log_path.read_text(errors="ignore").splitlines():
        if line.startswith("Epoch "):
            if current:
                epoch_data.append(current)
            current = {"epoch": int(line.split()[1])}

        elif "train_loss" in line:
            current["train_loss"] = float(line.split()[-1])

        elif "val_loss" in line:
            current["val_loss"] = float(line.split()[-1])

        elif "Pseudo dice" in line:
            nums = re.findall(r"[0-9]*\.?[0-9]+", line)
            if nums:
                current["pseudo_dice"] = sum(map(float, nums)) / len(nums)

    if current:
        epoch_data.append(current)

    return epoch_data


def set_nnunet_env_vars(env: dict[str, str], nnunet_raw: Path, nnunet_preprocessed: Path, nnunet_results: Path) -> None:
    env["nnUNet_raw"] = str(nnunet_raw)
    env["nnUNet_preprocessed"] = str(nnunet_preprocessed)
    env["nnUNet_results"] = str(nnunet_results)


def setup_logger(log_path: Path) -> None:
    logger.remove()  # remove default stderr logger

    # Console: user-facing info
    logger.add(sys.stdout, level="INFO")

    # File: full debug trace
    logger.add(log_path, level="DEBUG", rotation="100 MB")


def set_seed(seed: int) -> None:
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensures deterministic behavior (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def is_gcs_path(p: str) -> bool:
    return isinstance(p, str) and p.startswith("gs://")


def gcs_rsync_dir(local_dir: Path, gcs_dir: str, env: dict[str, str]) -> None:
    """
    Sync a local directory to a GCS directory.
    Requires 'gsutil' to be available inside the container.
    """
    if not local_dir.exists():
        raise FileNotFoundError(f"Local directory not found for upload: {local_dir}")

    cmd = ["gsutil", "-m", "rsync", "-r", str(local_dir), gcs_dir]
    logger.info(f"Uploading directory to GCS: {local_dir} -> {gcs_dir}")
    run_and_tee(cmd, (Path.cwd() / "gcs_upload.log"), env)


def gcs_cp_file(local_file: Path, gcs_path: str, env: dict[str, str]) -> None:
    """
    Copy a single local file to a GCS path (file or folder).
    Requires 'gsutil' in the container.
    """
    if not local_file.exists():
        raise FileNotFoundError(f"Local file not found for upload: {local_file}")

    cmd = ["gsutil", "cp", str(local_file), gcs_path]
    logger.info(f"Uploading file to GCS: {local_file} -> {gcs_path}")
    run_and_tee(cmd, (Path.cwd() / "gcs_upload.log"), env)



@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    project_root = Path(hydra.utils.get_original_cwd())

    # --- seed ---
    seed = int(cfg.reproducibility.seed)
    set_seed(seed)

    # --- read config ---
    dataset_id = int(cfg.dataset.dataset_id)
    nnunet_config = str(cfg.training.get("nnunet_config", "2d"))
    fold = str(cfg.training.get("fold", 0))
    trainer = str(cfg.training.get("trainer", "nnUNetTrainer"))
    device = str(cfg.training.get("device", "cpu"))
    epochs = cfg.training.get("epochs", None)

    # map epochs → nnU-Net trainer variant
    if epochs is not None and trainer.lower() in {"auto", "nnunettrainer"}:
        trainer = f"nnUNetTrainer_{int(epochs)}epochs"


    # --- paths (configurable, default to repo layout if missing) ---
    nnunet_raw = project_root / str(cfg.paths.get("nnunet_raw", "data/nnUNet_raw"))
    nnunet_preprocessed = project_root / str(cfg.paths.get("nnunet_preprocessed", "data/nnUNet_preprocessed"))
    nnunet_results = project_root / str(cfg.paths.get("nnunet_results", "data/nnUNet_results"))

    logs_dir = project_root / str(cfg.logging.get("logs_dir", "logs"))
    ensure_dir(logs_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"nnunet_{dataset_id}_{nnunet_config}_fold{fold}_{timestamp}.log"

    # --- setup logging ---
    setup_logger(log_path)
    wb_run = maybe_init_wandb(cfg)

    logger.info(f"Using seed: {seed}")
    logger.info("Initializing nnU-Net training")
    logger.info(
        f"Dataset={dataset_id}, config={nnunet_config}, fold={fold}, "
        f"trainer={trainer}, device={device}"
    )

    # --- env vars for nnUNet ---
    env = os.environ.copy()
    set_nnunet_env_vars(env, nnunet_raw, nnunet_preprocessed, nnunet_results)

    logger.info("nnU-Net environment variables set")
    logger.debug(f"nnUNet_raw={env['nnUNet_raw']}")
    logger.debug(f"nnUNet_preprocessed={env['nnUNet_preprocessed']}")
    logger.debug(f"nnUNet_results={env['nnUNet_results']}")

    # required dirs
    require_dir(nnunet_raw, "nnUNet_raw")
    require_dir(nnunet_preprocessed, "nnUNet_preprocessed")

    # --- run training ---
    start = time.time()
    try:
        cmd_train = [
            "nnUNetv2_train",
            str(dataset_id),
            nnunet_config,
            fold,
            "-tr",
            trainer,
            "-device",
            device,
        ]

        logger.info(f"Starting nnU-Net training. Logs will be written to: {log_path}")
        logger.info(f"Executing command: {' '.join(cmd_train)}")
        run_and_tee(cmd_train, log_path, env)

        duration_s = time.time() - start
        logger.success(f"Done! Training finished in {duration_s:.1f}s")
        logger.info(f"Log written to {log_path}")

        # upload to GCS (Vertex)
        gcs_out = cfg.paths.get("gcs_output", None)
        if gcs_out and is_gcs_path(str(gcs_out)):
            gcs_out = str(gcs_out).rstrip("/")

            # Upload nnU-Net results + logs
            gcs_rsync_dir(nnunet_results, f"{gcs_out}/nnUNet_results", env)
            gcs_cp_file(log_path, f"{gcs_out}/nnunet.log", env)
            logger.success(f"Uploaded artifacts to: {gcs_out}")


        if wb_run is not None:
            import wandb  # type: ignore

            epoch_metrics = parse_epoch_metrics(log_path)
            for m in epoch_metrics:
                wandb.log(m, step=m["epoch"])

            wb_run.finish()

    except Exception as e:
        logger.exception("Training failed")

        if wb_run is not None:
            try:
                import wandb  # type: ignore
                wandb.log({"failed": 1})
                wb_run.finish(exit_code=1)
            except Exception:
                pass
        raise



if __name__ == "__main__":
    main()
