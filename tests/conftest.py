import os
from pathlib import Path

import pytest

# ------------------------------------------------------------------
# Shared path fixtures
# ------------------------------------------------------------------

@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def data_root(project_root: Path) -> Path:
    return project_root / "data"


# ------------------------------------------------------------------
# Track which test files were executed
# ------------------------------------------------------------------

def pytest_sessionstart(session):
    session._executed_test_files = set()


def pytest_runtest_protocol(item, nextitem):
    """
    Called for each test item. Record the test file path.
    """
    item.session._executed_test_files.add(str(item.fspath))
    # Returning None lets pytest continue with normal execution
    return None

# ------------------------------------------------------------------
# Set nnUNet env vars for the test session
# ------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def set_nnunet_env_for_tests(project_root: Path):
    """
    Ensure nnUNet env vars are set for the pytest process.
    autouse=True -> runs automatically for the whole session.
    """
    data_dir = project_root / "data"

    os.environ.setdefault("nnUNet_raw", str(data_dir / "nnUNet_raw"))
    os.environ.setdefault("nnUNet_preprocessed", str(data_dir / "nnUNet_preprocessed"))
    os.environ.setdefault("nnUNet_results", str(data_dir / "nnUNet_results"))


# ------------------------------------------------------------------
# Print summary ONLY if all tests passed
# ------------------------------------------------------------------

def pytest_sessionfinish(session, exitstatus):
    if exitstatus != 0:
        return  # don't print anything if tests failed or error'd

    executed = session._executed_test_files

    print("\n ALL REQUESTED TESTS PASSED\n")

    if any("test_data.py" in f for f in executed):
        print("""Data tests passed successfully
            You have the expected directories
            and the same number of images and labels.""")

    if any("test_model.py" in f for f in executed):
        print("Model tests passed successfully")

    if any("test_training.py" in f for f in executed):
        print("Training tests passed successfully")

    print("")
