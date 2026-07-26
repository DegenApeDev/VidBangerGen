#!/usr/bin/env python3
"""Run VidBangerGen setup from a fresh source checkout."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apps.api.setup_wizard import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
