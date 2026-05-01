"""Helper to run the Streamlit app.

You can also run Streamlit directly:
- streamlit run src/app/streamlit_app.py

This script exists because beginners often like a single Python command.

Run it like:
- py scripts/run_app.py
"""

from __future__ import annotations

import subprocess
import sys


def main() -> None:
    cmd = [sys.executable, "-m", "streamlit", "run", "src/app/streamlit_app.py"]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
