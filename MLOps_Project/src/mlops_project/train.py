from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def repo_root() -> Path:
    """
    This file is: <repo>/src/mlops_project/train.py
    parents[0] = mlops_project
    parents[1] = src
    parents[2] = <repo>
    """
    return Path(__file__).resolve().parents[2]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_cfg(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError("config.yaml must contain a YAML mapping (dict) at the top level")
    return cfg


def run_and_tee(cmd: list[str], log_path: Path, env: dict[str, str]) -> None:
    """Run a command, stream stdout/stderr to terminal, and write everything to a log file."""
    with log_path.open("a", encoding="utf-8") as f:
        f.write("\n" + "=" * 120 + "\n")
        f.write(f"COMMAND: {' '.join(cmd)}\n")
        f.write("=" * 120 + "\n")

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )

        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            f.write(line)

        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"Command failed with exit code {rc}: {' '.join(cmd)}")


def maybe_init_wandb(cfg: dict[str, Any], log_path: Path) -> Optional[object]:
    wandb_cfg = cfg.get("wandb", {}) or {}
    enabled = bool(wandb_cfg.get("enabled", False))
    if not enabled:
        return None

    try:
        import wandb  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "wandb is enabled in config.yaml but not installed. Install it with:\n"
            "  pip install wandb\n"
        ) from e

    training = cfg.get("training", {}) or {}

    run = wandb.init(
        project=wandb_cfg.get("project", "mlops-nnunet"),
        entity=wandb_cfg.get("entity", None),
        name=wandb_cfg.get("run_name", None),
        config={
            # log exactly what we train with
            "dataset_id": training.get("dataset_id"),
            "config": training.get("nnunet_config"),
            "fold": training.get("fold"),
            "trainer": training.get("trainer"),
            "device": training.get("device"),
        },
    )

    wandb.save(str(log_path), policy="now")
    return run


def main() -> None:
    root = repo_root()

    # Optional CLI override for config path (but no need to pass dataset/config/fold etc.)
    parser = argparse.ArgumentParser(description="nnU-Net v2 training runner (train-only, config-driven)")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml (relative to repo)")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (root / config_path).resolve()

    cfg = load_cfg(config_path)

    # ---- Read training args from config ----
    training = cfg.get("training", {}) or {}

    if "dataset_id" not in training:
        raise KeyError("Missing required key: training.dataset_id in config.yaml")


    dataset_id = training["dataset_id"]          # required
    nnunet_config = training.get("nnunet_config", "2d")
    fold = str(training.get("fold", 0))
    trainer = training.get("trainer", "nnUNetTrainer")
    device = training.get("device", "cpu")

    # ---- Paths (also from config, default to repo layout) ----
    paths = cfg.get("paths", {}) or {}
    nnunet_raw = root / paths.get("nnunet_raw", "data/nnUNet_raw")
    nnunet_preprocessed = root / paths.get("nnunet_preprocessed", "data/nnUNet_preprocessed")
    nnunet_results = root / paths.get("nnunet_results", "data/nnUNet_results")

    logging_cfg = cfg.get("logging", {}) or {}
    logs_dir = root / logging_cfg.get("logs_dir", "logs")
    ensure_dir(logs_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"nnunet_{dataset_id}_{nnunet_config}_fold{fold}_{timestamp}.log"


    # Ensure required directories exist
    def require_dir(path: Path, name: str) -> None:
        if not path.exists() or not path.is_dir():
            raise FileNotFoundError(
                f"{name} directory is missing:\n  {path}\n"
                "This directory must already exist. "
                "It will appear only if the corresponding nnU-Net step ran successfully."
            )


    # nnU-Net expects these env vars
    env = os.environ.copy()
    env["nnUNet_raw"] = str(nnunet_raw)
    env["nnUNet_preprocessed"] = str(nnunet_preprocessed)
    env["nnUNet_results"] = str(nnunet_results)

    # Required inputs must exist
    require_dir(nnunet_raw, "nnUNet_raw")
    require_dir(nnunet_preprocessed, "nnUNet_preprocessed")

    # optional W&B
    wb_run = maybe_init_wandb(cfg, log_path)

    start = time.time()
    try:
        # ---- TRAIN ONLY ----
        cmd_train = [
            "nnUNetv2_train",
            str(dataset_id),
            str(nnunet_config),
            str(fold),
            "-tr",
            str(trainer),
            "-device",
            str(device),
        ]
        run_and_tee(cmd_train, log_path, env)

        duration_s = time.time() - start
        print(f"\n Done. Total duration: {duration_s:.1f}s")
        print(f"Log written to: {log_path}")

        if wb_run is not None:
            import wandb  # type: ignore

            wandb.log({"duration_seconds": duration_s})
            wandb.save(str(log_path), policy="now")
            wb_run.finish()

    except Exception:
        if wb_run is not None:
            try:
                import wandb  # type: ignore

                wandb.log({"failed": 1})
                wandb.save(str(log_path), policy="now")
                wb_run.finish(exit_code=1)
            except Exception:
                pass
        raise


if __name__ == "__main__":
    main()
