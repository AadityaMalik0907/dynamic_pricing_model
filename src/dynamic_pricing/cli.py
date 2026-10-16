"""Command-line entry points for local development."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .data.demo import make_demo_observations
from .models import train_forecaster, train_hotel_arrivals


def main() -> None:
    parser = argparse.ArgumentParser(prog="dynamic-pricing")
    parser.add_argument("command", choices=("api", "dashboard", "train-demo", "train-hotel"))
    parser.add_argument("--domain", choices=("hotel", "ecommerce"), default="hotel")
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--days", type=int, default=120)
    parser.add_argument("--data", help="path to hotel_bookings.csv for train-hotel")
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
    elif args.command == "train-demo":
        result = train_forecaster(
            make_demo_observations(domain=args.domain, days=args.days),
            domain=args.domain,
            artifact_dir=args.artifact_dir,
        )
        print(json.dumps(result.__dict__, indent=2))
    else:
        if not args.data:
            parser.error("train-hotel requires --data PATH")
        result = train_hotel_arrivals(args.data, artifact_dir=args.artifact_dir)
        print(json.dumps(result.__dict__, indent=2))
