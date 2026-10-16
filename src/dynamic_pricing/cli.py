"""Command-line entry points for local development."""

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(prog="dynamic-pricing")
    parser.add_argument("command", choices=("api", "dashboard"))
    args = parser.parse_args()

    if args.command == "api":
        result = subprocess.run(
            [sys.executable, "-m", "uvicorn", "dynamic_pricing.api.main:app", "--reload"],
            check=False,
        )
        raise SystemExit(result.returncode)
    elif args.command == "dashboard":
        dashboard = Path(__file__).parent / "dashboard" / "app.py"
        result = subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(dashboard)],
            check=False,
        )
        raise SystemExit(result.returncode)
