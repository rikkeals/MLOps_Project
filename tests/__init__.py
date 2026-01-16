from pathlib import Path

TEST_ROOT = Path(__file__).resolve().parent          # <repo>/tests
PROJECT_ROOT = TEST_ROOT.parent                     # <repo>
SRC_ROOT = PROJECT_ROOT / "src"                     # <repo>/src
DATA_ROOT = PROJECT_ROOT / "data"                   # <repo>/data

