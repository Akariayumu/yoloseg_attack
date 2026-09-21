import pytest

torch = pytest.importorskip("torch")

from yolo_mask_attack.attack.objectives import joint_attack_objective
from yolo_mask_attack.attack.proxy import locate_frozen_candidate
from yolo_mask_attack.models.types import DecodedBatch


def test_locate_frozen_candidate_uses_level_and_grid() -> None:
    decoded = DecodedBatch(
        boxes_xywh=torch.zeros(1, 3, 4),
        boxes_xyxy=torch.zeros(1, 3, 4),
        class_logits=torch.zeros(1, 3, 2),
        scores=torch.zeros(1, 3, 2),
        coefficients=torch.zeros(1, 3, 4),
        levels=torch.tensor([0, 0, 1]),
        grid_yx=torch.tensor([[2, 3], [2, 4], [2, 3]]),
    )
    assert locate_frozen_candidate(decoded, 1, (2, 3)) == 2


def test_joint_objective_has_expected_weights_and_gradients() -> None:
    mask_iou = torch.tensor([0.7], requires_grad=True)
    confidence = torch.tensor([0.8], requires_grad=True)
    box_iou = torch.tensor([0.9], requires_grad=True)
    loss = joint_attack_objective(
        mask_iou, confidence, box_iou, confidence_weight=2.0, box_weight=3.0
    )
    loss.backward()
    assert loss.item() == pytest.approx(5.0)
    assert mask_iou.grad.item() == pytest.approx(1.0)
    assert confidence.grad.item() == pytest.approx(2.0)
    assert box_iou.grad.item() == pytest.approx(3.0)
