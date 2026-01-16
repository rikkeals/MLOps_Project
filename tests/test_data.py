import os
from sys import path
from torch.utils.data import Dataset
from pathlib import Path
import pytest
import yaml

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
# Tests
#####################################################################
def test_nnunet_raw_ready(data_root, cfg):
    path = data_root / "nnUNet_raw"
    assert path.exists(), "nnUNet_raw missing - Have you run the data.py script yet?"
    assert (path / cfg["dataset"]["name"]).exists(), "Dataset folder missing - Option 1: Did you download the data yet by running the data.py script? Option 2: Is dataset_name correct in config.yaml? If you change the name in config after download; they will mismatch."

def test_nnunet_raw_length(data_root, cfg):
    path_images = data_root / "nnUNet_raw" / cfg["dataset"]["name"] / "imagesTr"
    images = list(path_images.glob("*.nii.gz"))
    path_labels = data_root / "nnUNet_raw" / cfg["dataset"]["name"] / "labelsTr"
    labels = list(path_labels.glob("*.nii.gz"))

    assert len(images) > 0, "No training images found in nnUNet_raw/imagesTr - Have you run the data.py script yet?"
    assert len(labels) > 0, "No training labels found in nnUNet_raw/labelsTr - Have you run the data.py script yet?"
    assert len(images) == len(labels), "Number of images and labels do not match in nnUNet_raw - Is there a file starting with '._' in imagesTr or labelsTr? Then this should be deleted."


def test_enviroment(data_root, cfg):
    nnunet_raw = data_root / "nnUNet_raw"
    nnunet_preprocessed = data_root / "nnUNet_preprocessed"
    nnunet_results = data_root / "nnUNet_results"

    assert os.environ.get("nnUNet_raw") == str(nnunet_raw), "nnUNet_raw env var not set correctly"
    assert os.environ.get("nnUNet_preprocessed") == str(nnunet_preprocessed), "nnUNet_preprocessed env var not set correctly"
    assert os.environ.get("nnUNet_results") == str(nnunet_results), "nnUNet_results env var not set correctly"






