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
