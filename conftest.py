"""Make the top-level ``sim`` package importable from tests.

The simulator is deliberately not shipped inside the installed package: it
exists to test the app, not to be part of it.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
