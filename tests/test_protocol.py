from yolo_mask_attack.eval.match import Outcome, classify_outcome
from yolo_mask_attack.eval.metrics import outcome_rates


def test_outcomes_are_mutually_exclusive_and_rates_sum_to_one() -> None:
    cases = [
        {"matched": False, "class_preserved": False, "confidence_preserved": False, "box_iou": 0, "mask_iou": 0},
        {"matched": True, "class_preserved": False, "confidence_preserved": True, "box_iou": 1, "mask_iou": 0},
        {"matched": True, "class_preserved": True, "confidence_preserved": True, "box_iou": 0.5, "mask_iou": 0},
        {"matched": True, "class_preserved": True, "confidence_preserved": True, "box_iou": 1, "mask_iou": 0.2},
        {"matched": True, "class_preserved": True, "confidence_preserved": True, "box_iou": 1, "mask_iou": 0.8},
    ]
    outcomes = [
        classify_outcome(**case, box_iou_min=0.9, mask_iou_max=0.5) for case in cases
    ]
    assert set(outcomes) == set(Outcome)
    rates = outcome_rates(outcomes)
    assert sum(rates[outcome.value] for outcome in Outcome) == 1.0
    assert rates["sdr"] == 0.2
    assert rates["dpr"] == 0.4
