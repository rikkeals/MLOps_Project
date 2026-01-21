import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import FileResponse
from omegaconf import OmegaConf

print("=== API STARTUP CHECK ===")
print("CWD:", os.getcwd())
print("PYTHONPATH:", os.environ.get("PYTHONPATH"))
print("Files in /app:", list(Path("/app").iterdir()) if Path("/app").exists() else "NO /app")
print("Files in /app/src:", list(Path("/app/src").iterdir()) if Path("/app/src").exists() else "NO /app/src")
print("Files in /app/configs:", list(Path("/app/configs").iterdir()) if Path("/app/configs").exists() else "NO /app/configs")
print("=========================")


# Load project config
CONFIG_PATH = Path("configs/config.yaml")
if not CONFIG_PATH.exists():
    raise RuntimeError(f"Config file not found: {CONFIG_PATH}")

cfg = OmegaConf.load(CONFIG_PATH)

DATASET_ID = int(cfg.dataset.dataset_id)          # 621
DATASET_NAME = cfg.dataset.name                   # Dataset621_Hippocampus
CONFIG = cfg.training.nnunet_config               # 2d
FOLD = str(cfg.training.fold)                     # 0
TRAINER = cfg.training.trainer                    # nnUNetTrainer
PLANS = cfg.training.plans                        # nnUNetPlans
DEFAULT_DEVICE = cfg.training.device              # cpu

REPO_ROOT = Path.cwd()

NNUNET_RAW = (REPO_ROOT / cfg.paths.nnunet_raw).resolve()
NNUNET_PRE = (REPO_ROOT / cfg.paths.nnunet_preprocessed).resolve()
NNUNET_RES = (REPO_ROOT / cfg.paths.nnunet_results).resolve()

# Set nnU-Net environment variables

os.environ["nnUNet_raw"] = str(NNUNET_RAW)
os.environ["nnUNet_preprocessed"] = str(NNUNET_PRE)
os.environ["nnUNet_results"] = str(NNUNET_RES)

NNUNET_RAW.mkdir(parents=True, exist_ok=True)
NNUNET_PRE.mkdir(parents=True, exist_ok=True)
NNUNET_RES.mkdir(parents=True, exist_ok=True)


def find_checkpoint() -> Optional[Path]:
    ckpts = list(NNUNET_RES.rglob("checkpoint_final.pth"))
    return ckpts[0] if ckpts else None



# FastAPI app

app = FastAPI(title="nnU-Net Inference API")


@app.get("/")
def health():
    ckpt = find_checkpoint()
    return {
        "status": "ok",
        "dataset": DATASET_NAME,
        "dataset_id": DATASET_ID,
        "config": CONFIG,
        "fold": FOLD,
        "trainer": TRAINER,
        "plans": PLANS,
        "default_device": DEFAULT_DEVICE,
        "model_ready": ckpt is not None,
        "nnUNet_results": str(NNUNET_RES),
    }


def run_nnunet_predict(input_dir: Path, output_dir: Path, device: str):
    cmd = [
        "nnUNetv2_predict",
        "-i", str(input_dir),
        "-o", str(output_dir),
        "-d", str(DATASET_ID),
        "-c", CONFIG,
        "-f", FOLD,
        "-tr", TRAINER,
        "-p", PLANS,
        "-device", device,
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "nnUNetv2_predict failed\n"
            f"CMD: {' '.join(cmd)}\n\n"
            f"STDOUT:\n{proc.stdout}\n\n"
            f"STDERR:\n{proc.stderr}"
        )


@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    device: str = Query(DEFAULT_DEVICE, description="cpu / cuda / mps"),
):
    ckpt = find_checkpoint()
    if ckpt is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model not trained yet. "
                "Run nnU-Net training to create checkpoint_final.pth "
                "under data/nnUNet_results."
            ),
        )

    if not image.filename.endswith((".nii", ".nii.gz")):
        raise HTTPException(status_code=400, detail="Upload a .nii or .nii.gz file")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        in_dir = tmp / "input"
        out_dir = tmp / "output"
        in_dir.mkdir()
        out_dir.mkdir()

        # single-channel input
        case_file = in_dir / "case_0000.nii.gz"
        case_file.write_bytes(await image.read())

        try:
            run_nnunet_predict(in_dir, out_dir, device)
        except RuntimeError as e:
            raise HTTPException(status_code=500, detail=str(e))

        pred = out_dir / "case.nii.gz"
        if not pred.exists():
            raise HTTPException(status_code=500, detail="Prediction failed")

        return FileResponse(
            pred,
            media_type="application/gzip",
            filename="prediction_case.nii.gz",
        )
