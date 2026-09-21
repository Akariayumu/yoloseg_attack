#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from yolo_mask_attack.attack.objectives import Method
from yolo_mask_attack.eval.aggregate import aggregate_attack_records


def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return records


def write_summary(records: list[dict[str, Any]], path: Path) -> None:
    summary = aggregate_attack_records(records)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-set", type=Path, default=Path("outputs/reference_set_v1.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/stage2_batch"))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=[method.value for method in Method],
        default=[Method.FIXED_WEIGHT.value, Method.DYNAMIC_WEIGHT.value, Method.CONSTRAINED.value],
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--retry-failures", action="store_true")
    args = parser.parse_args()

    reference_payload = json.loads(args.reference_set.read_text(encoding="utf-8"))
    total_references = len(reference_payload["instances"])
    stop = min(args.start + args.limit, total_references)
    if not 0 <= args.start < stop:
        raise ValueError("Requested reference range is empty or out of bounds")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.output_dir / "records.jsonl"
    summary_path = args.output_dir / "summary.json"
    records = load_records(records_path)
    latest = {int(record["reference_index"]): record for record in records}
    runner = Path(__file__).with_name("run_stage2_smoke.py")

    for reference_index in range(args.start, stop):
        previous = latest.get(reference_index)
        if previous and (previous.get("status") == "completed" or not args.retry_failures):
            print(f"skip reference={reference_index} status={previous.get('status')}")
            continue
        instance_dir = args.output_dir / "instances" / f"{reference_index:05d}"
        command = [
            sys.executable,
            str(runner),
            "--reference-set",
            str(args.reference_set),
            "--reference-index",
            str(reference_index),
            "--output-dir",
            str(instance_dir),
            "--device",
            args.device,
            "--steps",
            str(args.steps),
            "--methods",
            *args.methods,
        ]
        started = time.perf_counter()
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        if completed.returncode == 0:
            metrics = json.loads((instance_dir / "metrics.json").read_text(encoding="utf-8"))
            record: dict[str, Any] = {
                "status": "completed",
                "reference_index": reference_index,
                "wall_seconds": time.perf_counter() - started,
                "metrics": metrics,
            }
        else:
            record = {
                "status": "failed",
                "reference_index": reference_index,
                "wall_seconds": time.perf_counter() - started,
                "returncode": completed.returncode,
                "error": (completed.stderr or completed.stdout)[-4000:],
            }
        with records_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        records.append(record)
        latest[reference_index] = record
        write_summary(list(latest.values()), summary_path)
        print(f"reference={reference_index} status={record['status']}")

    write_summary(list(latest.values()), summary_path)
    print(f"records={records_path} summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
