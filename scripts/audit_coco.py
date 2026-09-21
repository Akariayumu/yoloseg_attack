#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from yolo_mask_attack.config import load_config
from yolo_mask_attack.data import CocoPersonDataset
from yolo_mask_attack.utils.hash import sha256_file


def build_report(config_path: Path, sample_size: int) -> dict[str, object]:
    config = load_config(config_path)
    dataset_config = config["dataset"]
    dataset = CocoPersonDataset(dataset_config["images"], dataset_config["annotations"])
    records = dataset.records(limit=sample_size, seed=20260921)
    missing = dataset.validate_files()
    sample_instances = sum(len(record.annotation_ids) for record in records)
    sample_mask_pixels = sum(int(dataset.masks(record).sum()) for record in records)
    return {
        "dataset": dataset_config["name"],
        "images_with_person": len(dataset),
        "missing_image_files": len(missing),
        "annotation_sha256": sha256_file(dataset.annotations),
        "sample_size": len(records),
        "sample_instances": sample_instances,
        "sample_mask_pixels": sample_mask_pixels,
        "sample_image_ids": [record.image_id for record in records],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/stage1_protocol.yaml"))
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("outputs/coco_audit.json"))
    args = parser.parse_args()
    report = build_report(args.config, args.sample_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
