from __future__ import annotations

import argparse
import importlib.util
import platform
from pathlib import Path

from .config import load_config, require_keys


def doctor(config_path: str) -> int:
    config = load_config(config_path)
    require_keys(config, "experiment", "model", "attack", "constraints")
    print(f"python={platform.python_version()}")
    for package in ("torch", "torchvision", "ultralytics", "numpy", "yaml"):
        status = "installed" if importlib.util.find_spec(package) else "missing"
        print(f"{package}={status}")
    weights = Path(config["model"]["weights"])
    print(f"weights={weights} ({'found' if weights.is_file() else 'missing'})")
    print(f"config={Path(config_path).resolve()} (valid)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="yolo-mask-attack")
    subparsers = parser.add_subparsers(dest="command", required=True)
    doctor_parser = subparsers.add_parser("doctor", help="check configuration and dependencies")
    doctor_parser.add_argument("--config", default="configs/base.yaml")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        return doctor(args.config)
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
