from yolo_mask_attack.eval.aggregate import aggregate_attack_records


def _method(outcome: str, mask_iou: float) -> dict:
    return {
        "elapsed_seconds": 1.0,
        "official": {
            "outcome": outcome,
            "adversarial_confidence": 0.7,
            "box_iou": 0.95,
            "mask_iou": mask_iou,
        },
    }


def test_aggregate_attack_records_computes_sdr_and_dpr() -> None:
    records = [
        {
            "status": "completed",
            "reference_index": 0,
            "metrics": {"methods": {"fixed_weight": _method("selective_degradation", 0.2)}},
        },
        {
            "status": "completed",
            "reference_index": 1,
            "metrics": {"methods": {"fixed_weight": _method("preserved_not_degraded", 0.8)}},
        },
        {"status": "failed", "reference_index": 2},
    ]
    summary = aggregate_attack_records(records)
    fixed = summary["methods"]["fixed_weight"]
    assert summary["completed_references"] == 2
    assert summary["failed_records"] == 1
    assert fixed["sdr"] == 0.5
    assert fixed["dpr"] == 1.0
    assert fixed["mean_mask_iou"] == 0.5
