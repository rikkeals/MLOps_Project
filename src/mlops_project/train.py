from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import hydra
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
            env=env,
        )

        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            f.write(line)

        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"Command failed with exit code {rc}: {' '.join(cmd)}")


def maybe_init_wandb(cfg: DictConfig, log_path: Path) -> Optional[object]:
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
        project=str(cfg.wandb.get("project", "mlops-nnunet")),
        entity=cfg.wandb.get("entity", None),
        name=cfg.wandb.get("run_name", None),
        config={
            "dataset_id": int(cfg.dataset.dataset_id),
            "nnunet_config": str(cfg.training.get("nnunet_config", "2d")),
            "fold": str(cfg.training.get("fold", 0)),
            "trainer": str(cfg.training.get("trainer", "nnUNetTrainer")),
            "device": str(cfg.training.get("device", "cpu")),
        },
    )

    wandb.save(str(log_path), policy="now")
    return run


def set_nnunet_env_vars(env: dict[str, str], nnunet_raw: Path, nnunet_preprocessed: Path, nnunet_results: Path) -> None:
    env["nnUNet_raw"] = str(nnunet_raw)
    env["nnUNet_preprocessed"] = str(nnunet_preprocessed)
    env["nnUNet_results"] = str(nnunet_results)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    project_root = Path(hydra.utils.get_original_cwd())

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

    # --- env vars for nnUNet ---
    env = os.environ.copy()
    set_nnunet_env_vars(env, nnunet_raw, nnunet_preprocessed, nnunet_results)

    print("nnU-Net environment variables set:")
    print("  nnUNet_raw:", env["nnUNet_raw"])
    print("  nnUNet_preprocessed:", env["nnUNet_preprocessed"])
    print("  nnUNet_results:", env["nnUNet_results"])

    # required dirs
    require_dir(nnunet_raw, "nnUNet_raw")
    require_dir(nnunet_preprocessed, "nnUNet_preprocessed")

    wb_run = maybe_init_wandb(cfg, log_path)

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
        print(f"Starting nnU-Net training. Logs will be written to: {log_path}\n")
        print(" ".join(cmd_train))
        subprocess.run(cmd_train, check=True, env=env)
        run_and_tee(cmd_train, log_path, env)

        duration_s = time.time() - start
        print(f"\nDone. Total duration: {duration_s:.1f}s")
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
