import os
from pathlib import Path

import pytest


@pytest.fixture
def catalog_dir() -> Path:
    return Path(os.environ["CATALOG_DIR"])
