from pathlib import Path
from typing import Any
import gdown
import hydra
from omegaconf import DictConfig
import tarfile
import json
import shutil


"""
This script downloads and preprocesses the Hippocampus dataset
from Medical Segmentation Decathlon for use with nnU-Net v2.
"""


################################################################
# Helpers
################################################################

def _to_0000_name(name: str) -> str:
    """hippocampus_001.nii.gz -> hippocampus_001_0000.nii.gz"""
    if name.endswith("_0000.nii.gz"):
        return name
    return name[:-7] + "_0000.nii.gz" if name.endswith(".nii.gz") else name


def _is_junk_file(p: Path) -> bool:
    """Filter common junk files created by macOS/Windows tooling."""
    return p.name.startswith("._") or p.name in {".DS_Store", "Thumbs.db"}


################################################################
# Download + extract
################################################################

def download_and_extract_dataset(
    data_dir: Path,
    hippocampus_id: str,
) -> Path:
    """
    Download and extract the Hippocampus dataset into data_dir.
    Returns the extracted folder path (Task04_Hippocampus).
    """
    archive_path = data_dir / "Task_04_hippocampus.tar"
    extracted_dir = data_dir / "Task04_Hippocampus"

    data_dir.mkdir(parents=True, exist_ok=True)

    if not extracted_dir.exists():
        print("Downloading hippocampus dataset from Google Drive...")

        gdown.download(
            id=hippocampus_id,
            output=str(archive_path),
            quiet=False,
            fuzzy=True,
        )

        print("Extracting dataset...")
        with tarfile.open(archive_path, "r:*") as tar:
            tar.extractall(data_dir)

        print(f"Dataset downloaded and extracted to: {extracted_dir}")
    else:
        print(f"Dataset already exists at: {extracted_dir}")

    return extracted_dir


################################################################
# Preprocessing to nnU-Net v2 format
################################################################

def copy_images_with_0000(src_task_dir: Path, dst_dataset_dir: Path) -> None:
    for split in ["imagesTr", "imagesTs"]:
        src = src_task_dir / split
        if not src.exists():
            continue

        dst = dst_dataset_dir / split
        dst.mkdir(parents=True, exist_ok=True)

        for f in src.glob("*.nii.gz"):
            if _is_junk_file(f):
                continue
            shutil.copy2(f, dst / _to_0000_name(f.name))


def copy_labels(src_task_dir: Path, dst_dataset_dir: Path) -> None:
    src = src_task_dir / "labelsTr"
    dst = dst_dataset_dir / "labelsTr"
    dst.mkdir(parents=True, exist_ok=True)

    for f in src.glob("*.nii.gz"):
        if _is_junk_file(f):
            continue
        shutil.copy2(f, dst / f.name)  # labels stay without _0000


def write_nnunetv2_dataset_json(
    src_task_dir: Path,
    dst_dataset_dir: Path,
    *,
    channel_name: str = "T1",
) -> None:
    src_json = src_task_dir / "dataset.json"
    dst_json = dst_dataset_dir / "dataset.json"

    with src_json.open("r", encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)

    # nnU-Net v2 fields
    data["file_ending"] = ".nii.gz"
    data["channel_names"] = {"0": channel_name}

    # Convert labels if old format {"0":"background",...}
    labels = data.get("labels")
    if isinstance(labels, dict) and all(isinstance(k, str) and k.isdigit() for k in labels.keys()):
        data["labels"] = {name: int(idx) for idx, name in labels.items()}

    # v2 infers cases from folder contents
    data.pop("training", None)
    data.pop("test", None)

    # Optional: modality not needed in v2
    data.pop("modality", None)

    with dst_json.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        f.write("\n")


def remove_appledouble_files(dataset_dir: Path) -> int:
    """Deletes macOS AppleDouble files like ._foo* that may slip in."""
    removed = 0
    for sub in ["imagesTr", "imagesTs", "labelsTr", "labelsTs"]:
        p = dataset_dir / sub
        if not p.exists():
            continue
        for f in p.glob("._*"):
            f.unlink()
            removed += 1
    return removed


def preprocess_to_nnunetv2(
    extracted_task_dir: Path,     # Task04_Hippocampus
    nnunet_raw_root: Path,        # .../nnUNet_raw
    dataset_name: str,            # Dataset621_Hippocampus
    *,
    channel_name: str = "T1",
) -> Path:
    """
    Converts Task04_Hippocampus -> nnUNet_raw/DatasetXXX_Name (nnU-Net v2).
    Returns the destination dataset directory.
    """
    dst_dataset_dir = nnunet_raw_root / dataset_name
    dst_dataset_dir.mkdir(parents=True, exist_ok=True)

    print("Preprocessing dataset for nnU-Net v2...")
    copy_images_with_0000(extracted_task_dir, dst_dataset_dir)
    copy_labels(extracted_task_dir, dst_dataset_dir)
    write_nnunetv2_dataset_json(extracted_task_dir, dst_dataset_dir, channel_name=channel_name)

    removed = remove_appledouble_files(dst_dataset_dir)
    if removed:
        print(f"Removed {removed} AppleDouble files from the preprocessed dataset.")

    print(f"✓ Dataset written to: {dst_dataset_dir}")
    return dst_dataset_dir


################################################################
# Uses Hydra for configuration and runs the process
################################################################

@hydra.main(version_base=None, config_path="../../configs", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    # IMPORTANT: Hydra changes working dir; resolve paths from original project cwd
    project_root = Path(hydra.utils.get_original_cwd())

    dataset_name: str = cfg.dataset.name
    raw_root = project_root / cfg.dataset.raw_root          # e.g. data/original
    nnunet_raw_root = project_root / cfg.dataset.nnunet_raw_root  # e.g. data/nnUNet_raw
    channel_name: str = getattr(cfg.dataset, "channel_name", "T1")

    hippocampus_id: str = getattr(cfg.dataset, "gdrive_id", "1RzPB1_bqzQhlWvU-YGvZzhx2omcDh38C")

    # 1) download + extract
    extracted_dir = download_and_extract_dataset(raw_root, hippocampus_id)

    # 2) preprocess into nnU-Net v2 structure
    preprocess_to_nnunetv2(
        extracted_task_dir=extracted_dir,
        nnunet_raw_root=nnunet_raw_root,
        dataset_name=dataset_name,
        channel_name=channel_name,
    )

    print("All done!")


if __name__ == "__main__":
    main()
