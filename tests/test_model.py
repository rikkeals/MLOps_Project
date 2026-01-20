import pytest
import yaml
import json
from pathlib import Path
from omegaconf import OmegaConf

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
# Tests if config is updated from plans file
#####################################################################

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


################################################################
# Tests for model.py functions
################################################################

def test_update_config_from_plans_2d(tmp_path):
    project_root = tmp_path
    dataset_name = "Dataset621_Hippocampus"

    plans_dir = (
        project_root
        / "data"
        / "nnUNet_preprocessed"
        / dataset_name
    )
    plans_dir.mkdir(parents=True)

    plans = {
        "configurations": {
            "2d": {
                "batch_size": 8,
                "patch_size": [128, 128],
                "architecture": "UNet"
            }
        }
    }

    (plans_dir / "nnUNetPlans.json").write_text(json.dumps(plans))

    cfg_path = project_root / "config.yaml"
    cfg_path.write_text("model: {}")

    from mlops_project.model import update_config_from_plans_2d
    update_config_from_plans_2d(project_root, dataset_name, cfg_path)

    cfg = OmegaConf.load(cfg_path)
    assert cfg.model.configuration == "2d"
    assert cfg.model.defaults["batch_size"] == 8
