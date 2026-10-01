"""tests/conftest.py

Test configuration and environment setup for RSNA Knee Abnormality Detection.
Appends user site-packages to sys.path if not present to allow access to local tools like matplotlib.
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure local user site packages can be discovered as fallback (e.g. for matplotlib)
user_site = Path.home() / ".local/lib/python3.13/site-packages"
if user_site.is_dir() and str(user_site) not in sys.path:
    sys.path.append(str(user_site))
