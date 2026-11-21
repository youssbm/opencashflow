"""Download the OpenCashFlow benchmark datasets.

Run from the repository root::

    python data/download.py --list
    python data/download.py --dataset polish german bangladesh_sme
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running this script without installing the package first.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from opencashflow.data.download import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
