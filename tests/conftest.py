"""Pytest conftest -- adds tests/ to sys.path for helper imports."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
