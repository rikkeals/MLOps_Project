import pytest
import yaml
import os
from pathlib import Path


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

def test_nnunet_env_vars_exist():
    for var in [
        "nnUNet_raw",
        "nnUNet_preprocessed",
        "nnUNet_results",
    ]:
        assert var in os.environ, f"{var} is not set in the shell environment"


def test_config_update_from_plans_2d(project_root: Path, cfg):
    config_path = project_root / "configs" / "config.yaml"
    dataset_name = cfg["dataset"]["name"]

    # Load the updated config
    with open(config_path, "r", encoding="utf-8") as f:
        updated_cfg = yaml.safe_load(f)

    # Check that the model configuration is set to 2d
    assert updated_cfg["model"]["configuration"] == "2d", "Model configuration is not set to 2d"

    # Check that defaults contain expected keys
    defaults = updated_cfg["model"]["defaults"]
    expected_keys = [
        "data_identifier",
        "batch_size",
        "patch_size",
        "spacing",
        "normalization_schemes",
        "architecture",
    ]

    for key in expected_keys:
        assert key in defaults, f"{key} not found in model.defaults - Have you run model.py script?"