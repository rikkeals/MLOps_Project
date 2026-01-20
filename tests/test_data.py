import pytest
import yaml
import json
from pathlib import Path
from mlops_project.data import _to_0000_name


####################################################################
# Fixtures
####################################################################
@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]

@pytest.fixture(scope="session")
def data_root(project_root: Path) -> Path:
    return project_root / "data"

@pytest.fixture(scope="session")
def cfg(project_root):
    config_path = project_root / "configs" / "config.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


#####################################################################
# Tests if data and directories are correctly set up
#####################################################################
@pytest.mark.integration
def test_nnunet_raw_ready(data_root, cfg):
    path = data_root / "nnUNet_raw"
    assert path.exists(), "nnUNet_raw missing - Have you run the data.py script yet?"
    assert (path / cfg["dataset"]["name"]).exists(), (
        "Dataset folder missing. Option 1: Did you download the data by running "
        "data.py? Option 2: Is dataset_name correct in config.yaml? If you change "
        "the name in config after download, they will mismatch."
    )

@pytest.mark.integration
def test_nnunet_raw_length(data_root, cfg):
    path_images = data_root / "nnUNet_raw" / cfg["dataset"]["name"] / "imagesTr"
    images = list(path_images.glob("*.nii.gz"))
    path_labels = data_root / "nnUNet_raw" / cfg["dataset"]["name"] / "labelsTr"
    labels = list(path_labels.glob("*.nii.gz"))

    assert len(images) > 0, "No training images found in nnUNet_raw/imagesTr - Have you run the data.py script yet?"
    assert len(labels) > 0, "No training labels found in nnUNet_raw/labelsTr - Have you run the data.py script yet?"
    assert len(images) == len(labels), (
        "Number of images and labels do not match in nnUNet_raw. "
        "Is there a file starting with '._' in imagesTr or labelsTr? "
        "If so, delete it."
    )


###############################################################
# Tests for data preprocessing functions
###############################################################

def test_to_0000_name_adds_suffix():
    assert _to_0000_name("hippo_001.nii.gz") == "hippo_001_0000.nii.gz"

def test_to_0000_name_idempotent():
    assert _to_0000_name("hippo_001_0000.nii.gz") == "hippo_001_0000.nii.gz"

def test_copy_images_with_0000(tmp_path):
    src = tmp_path / "Task04_Hippocampus" / "imagesTr"
    dst = tmp_path / "nnUNet_raw" / "Dataset621_Hippocampus"

    src.mkdir(parents=True)
    dst.mkdir(parents=True)

    img = src / "hippo_001.nii.gz"
    img.write_text("fake")

    from mlops_project.data import copy_images_with_0000
    copy_images_with_0000(src.parent, dst)

    copied = dst / "imagesTr" / "hippo_001_0000.nii.gz"
    assert copied.exists()


def test_copy_labels(tmp_path):
    src = tmp_path / "Task04_Hippocampus" / "labelsTr"
    dst = tmp_path / "nnUNet_raw" / "Dataset621_Hippocampus"

    src.mkdir(parents=True)
    dst.mkdir(parents=True)

    label = src / "hippo_001.nii.gz"
    label.write_text("label")

    from mlops_project.data import copy_labels
    copy_labels(src.parent, dst)

    assert (dst / "labelsTr" / "hippo_001.nii.gz").exists()


def test_write_nnunetv2_dataset_json(tmp_path):
    src = tmp_path / "Task04_Hippocampus"
    dst = tmp_path / "Dataset621_Hippocampus"

    src.mkdir()
    dst.mkdir()

    dataset_json = {
        "labels": {"0": "background", "1": "hippocampus"},
        "modality": {"0": "MR"},
        "training": [],
        "test": []
    }

    (src / "dataset.json").write_text(json.dumps(dataset_json))

    from mlops_project.data import write_nnunetv2_dataset_json
    write_nnunetv2_dataset_json(src, dst, channel_name="T1")

    out = json.loads((dst / "dataset.json").read_text())

    assert out["file_ending"] == ".nii.gz"
    assert out["channel_names"] == {"0": "T1"}
    assert out["labels"]["hippocampus"] == 1
    assert "training" not in out

def test_preprocess_to_nnunetv2(tmp_path):
    extracted = tmp_path / "Task04_Hippocampus"
    raw = tmp_path / "nnUNet_raw"

    (extracted / "imagesTr").mkdir(parents=True)
    (extracted / "labelsTr").mkdir()

    (extracted / "imagesTr" / "img.nii.gz").write_text("img")
    (extracted / "labelsTr" / "lbl.nii.gz").write_text("lbl")

    (extracted / "dataset.json").write_text(
        json.dumps({"labels": {"0": "bg", "1": "hippo"}})
    )

    from mlops_project.data import preprocess_to_nnunetv2
    dst = preprocess_to_nnunetv2(
        extracted_task_dir=extracted,
        nnunet_raw_root=raw,
        dataset_name="Dataset621_Hippocampus",
    )

    assert (dst / "imagesTr").exists()
    assert (dst / "labelsTr").exists()
    assert (dst / "dataset.json").exists()










