import pytest

torch = pytest.importorskip("torch")

from yolo_mask_attack.eval.match import greedy_iou_match


def test_greedy_matching_is_one_to_one() -> None:
    references = torch.tensor([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 9.0, 9.0]])
    predictions = torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]])
    reference_indices, prediction_indices, ious = greedy_iou_match(
        references, predictions, iou_threshold=0.5
    )
    assert reference_indices.tolist() == [0]
    assert prediction_indices.tolist() == [0]
    assert ious.tolist() == [1.0]


def test_greedy_matching_handles_empty_predictions() -> None:
    references = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    predictions = torch.empty((0, 4))
    reference_indices, prediction_indices, ious = greedy_iou_match(references, predictions)
    assert reference_indices.numel() == prediction_indices.numel() == ious.numel() == 0
