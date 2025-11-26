"""Configuration for pytest."""

import sys
from pathlib import Path

# Add src directory to path so we can import fedboost_mvs
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))
