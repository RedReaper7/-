from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crop_yield_prediction.config import get_settings
from crop_yield_prediction.pipeline import train_project


def main() -> None:
    settings = get_settings()
    result = train_project(settings)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
