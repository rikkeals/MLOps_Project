from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional


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


def run_and_tee(cmd: list[str], log_path: Path, env: dict[str, str]) -> None:
    """
    Run a command, stream stdout/stderr to terminal, and write everything to a log file.
    """
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


def maybe_init_wandb(args: argparse.Namespace, log_path: Path) -> Optional[object]:
    if not args.wandb:
        return None

    try:
        import wandb  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "wandb is enabled but not installed. Install it with:\n"
            "  pip install wandb\n"
        ) from e

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        config={
            "dataset_id": args.dataset_id,
            "config": args.nnunet_config,
            "fold": args.fold,
            "trainer": args.trainer,
            "device": args.device,
            "plans": args.plans,
            "preprocess": args.preprocess,
            "num_processes": args.num_processes,
        },
    )

    # Save the log file as it grows (wandb will upload at the end; we also upload explicitly later)
    wandb.save(str(log_path), policy="now")
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description="nnU-Net v2 training runner (plan+preprocess + train)")

    # Core nnU-Net args (matches your CLI usage)
    parser.add_argument("dataset_id", type=int, help="nnU-Net dataset id, e.g. 621")
    parser.add_argument("--nnunet_config", default="3d_fullres", help="nnU-Net configuration, e.g. 3d_fullres")
    parser.add_argument("--fold", default="0", help="Fold index, e.g. 0 (or 'all' if you want)")
    parser.add_argument("--trainer", default="nnUNetTrainer_5epochs", help="Trainer class name")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"], help="Training device")

    # Preprocessing control
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="If set, run nnUNetv2_plan_and_preprocess before training (recommended).",
    )
    parser.add_argument("--plans", default=None, help="Optional: plans identifier (rarely needed)")
    parser.add_argument(
        "--num_processes",
        type=int,
        default=8,
        help="Processes for preprocessing (nnU-Net flag: -np).",
    )

    # Override nnU-Net env paths (defaults to your repo ./data/*)
    parser.add_argument("--nnunet_raw", default=None, help="Override nnUNet_raw path")
    parser.add_argument("--nnunet_preprocessed", default=None, help="Override nnUNet_preprocessed path")
    parser.add_argument("--nnunet_results", default=None, help="Override nnUNet_results path")

    # Logging
    parser.add_argument("--logs_dir", default=None, help="Where to write logs (default: <repo>/logs)")

    # Weights & Biases
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", default="mlops-nnunet", help="W&B project name")
    parser.add_argument("--wandb_entity", default=None, help="W&B entity/team (optional)")
    parser.add_argument("--wandb_run_name", default=None, help="W&B run name (optional)")

    args = parser.parse_args()

    root = repo_root()

    # Default nnU-Net paths to your repo layout (like in your screenshot)
    nnunet_raw = Path(args.nnunet_raw) if args.nnunet_raw else root / "data" / "nnUNet_raw"
    nnunet_preprocessed = (
        Path(args.nnunet_preprocessed) if args.nnunet_preprocessed else root / "data" / "nnUNet_preprocessed"
    )
    nnunet_results = Path(args.nnunet_results) if args.nnunet_results else root / "data" / "nnUNet_results"

    logs_dir = Path(args.logs_dir) if args.logs_dir else root / "logs"
    ensure_dir(logs_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"nnunet_{args.dataset_id}_{args.nnunet_config}_fold{args.fold}_{timestamp}.log"

    # Prepare env for subprocesses
    env = os.environ.copy()
    env["nnUNet_raw"] = str(nnunet_raw)
    env["nnUNet_preprocessed"] = str(nnunet_preprocessed)
    env["nnUNet_results"] = str(nnunet_results)

    # Basic sanity: folders exist (create results/logs; raw/preprocessed should exist if you put them there)
    ensure_dir(nnunet_results)
    ensure_dir(nnunet_preprocessed)
    ensure_dir(nnunet_raw)

    # Init W&B (optional)
    wb_run = maybe_init_wandb(args, log_path)

    start = time.time()
    try:
        # 1) Plan + preprocess (if requested)
        if args.preprocess:
            cmd_pp = [
                "nnUNetv2_plan_and_preprocess",
                "-d",
                str(args.dataset_id),
                "-np",
                str(args.num_processes),
            ]
            if args.plans:
                cmd_pp += ["-pl", args.plans]  # only if you really need it
            run_and_tee(cmd_pp, log_path, env)

        # 2) Train
        cmd_train = [
            "nnUNetv2_train",
            str(args.dataset_id),
            args.nnunet_config,
            str(args.fold),
            "-tr",
            args.trainer,
            "-device",
            args.device,
        ]
        run_and_tee(cmd_train, log_path, env)

        duration_s = time.time() - start
        print(f"\n✅ Done. Total duration: {duration_s:.1f}s")
        print(f"Log written to: {log_path}")

        if wb_run is not None:
            import wandb  # type: ignore

            wandb.log({"duration_seconds": duration_s})
            wandb.save(str(log_path), policy="now")
            wb_run.finish()

    except Exception as e:
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
