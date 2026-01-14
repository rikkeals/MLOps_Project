from pathlib import Path
import gdown
import tarfile
import json
from typing import Any
import shutil


DATA_DIR = Path("data/original")
DATA_DIR.mkdir(parents=True, exist_ok=True)

HIPPOCAMPUS_ID = "1RzPB1_bqzQhlWvU-YGvZzhx2omcDh38C"
ARCHIVE_PATH = DATA_DIR / "Task_04_hippocampus.tar"
EXTRACTED_DIR = DATA_DIR / "Task04_Hippocampus"

DATASET_NAME = "Dataset621_Hippocampus"
PREPROCESSED_DATA_DIR = Path(f"data/nnUNet_raw/{DATASET_NAME}")

def download_and_extract_dataset() -> None:
    """
    Download and extract the Hippocampus dataset from Google Drive
    if it does not already exist in the data/original directory.
    """
    if not EXTRACTED_DIR.exists():
        print("Downloading hippocampus dataset from Google Drive...")

        gdown.download(
            id=HIPPOCAMPUS_ID,
            output=str(ARCHIVE_PATH),
            quiet=False,
            fuzzy=True
        )

        print("Extracting dataset...")
        with tarfile.open(ARCHIVE_PATH, "r:*") as tar:
            tar.extractall(DATA_DIR)

        print("Dataset downloaded and extracted.")
    else:
        print("Dataset already exists.")


def _to_0000_name(name: str) -> str:
    if name.endswith("_0000.nii.gz"):
        return name
    return name[:-7] + "_0000.nii.gz" if name.endswith(".nii.gz") else name


def copy_images_with_0000(src_task_dir: Path, dst_dataset_dir: Path) -> None:
    for split in ["imagesTr", "imagesTs"]:
        src = src_task_dir / split
        if not src.exists():
            continue

        dst = dst_dataset_dir / split
        dst.mkdir(parents=True, exist_ok=True)

        for f in src.glob("*.nii.gz"):
            shutil.copy2(f, dst / _to_0000_name(f.name))


def copy_labels(src_task_dir: Path, dst_dataset_dir: Path) -> None:
    src = src_task_dir / "labelsTr"
    dst = dst_dataset_dir / "labelsTr"
    dst.mkdir(parents=True, exist_ok=True)

    for f in src.glob("*.nii.gz"):
        shutil.copy2(f, dst / f.name)


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

    # Required nnU-Net v2 fields
    data["file_ending"] = ".nii.gz"
    data["channel_names"] = {"0": channel_name}

    # Convert labels to v2 format if needed
    labels = data.get("labels")
    if isinstance(labels, dict) and all(k.isdigit() for k in labels):
        data["labels"] = {name: int(idx) for idx, name in labels.items()}

    # Remove v1-only fields
    data.pop("training", None)
    data.pop("test", None)

    with dst_json.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        f.write("\n")


def remove_appledouble_files(dataset_dir: Path) -> int:
    """Deletes macOS AppleDouble files like ._foo.nii.gz in images/labels folders."""
    removed = 0
    for sub in ["imagesTr", "imagesTs", "labelsTr", "labelsTs"]:
        p = dataset_dir / sub
        if not p.exists():
            continue
        for f in p.glob("._*"):
            f.unlink()
            removed += 1
    return removed



def preprocess(
    data_path: Path,             # folder containing Task04_Hippocampus
    output_root: Path,            # nnUNet_raw (or nnUNet_raw_tester)
) -> None:
    """
    Converts Task04_Hippocampus → Dataset621_Hippocampus (nnU-Net v2).
    """
    src_task_dir = data_path 
    dst_dataset_dir = output_root / DATASET_NAME

    print("Preprocessing dataset for nnU-Net v2...")
    dst_dataset_dir.mkdir(parents=True, exist_ok=True)

    copy_images_with_0000(src_task_dir, dst_dataset_dir)
    copy_labels(src_task_dir, dst_dataset_dir)
    write_nnunetv2_dataset_json(src_task_dir, dst_dataset_dir)

    removed = remove_appledouble_files(PREPROCESSED_DATA_DIR)
    print(f"Removed {removed} AppleDouble files.")


    print(f"✓ Dataset written to: {dst_dataset_dir}")


if __name__ == "__main__":
    download_and_extract_dataset()
    preprocess(EXTRACTED_DIR, PREPROCESSED_DATA_DIR.parent)
    print("All done!")

# #Set environment variable for nnUNet
# $data = "C:\Users\rikke\OneDrive - Danmarks Tekniske Universitet\Universitet\Kandidat - MMC\Machine Learning Operations\MLOps_Project\MLOps_Project\data"

# $env:nnUNet_raw = "$data\nnUNet_raw"
# $env:nnUNet_preprocessed = "$data\nnUNet_preprocessed"
# $env:nnUNet_results = "$data\nnUNet_results"

