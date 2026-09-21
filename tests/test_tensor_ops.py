import pytest

torch = pytest.importorskip("torch")

from yolo_mask_attack.attack.base import apply_perturbation_mask, project_linf
from yolo_mask_attack.models.geometry import box_iou_aligned
from yolo_mask_attack.models.masks import hard_iou, instance_masks, soft_iou


def test_projection_respects_image_and_linf_bounds() -> None:
    image = torch.tensor([0.0, 0.5, 1.0])
    delta = torch.tensor([-1.0, 1.0, 1.0])
    projected = project_linf(delta, image, epsilon=0.1)
    assert torch.all(projected.abs() <= 0.100001)
    assert torch.all((image + projected >= 0) & (image + projected <= 1))


def test_perturbation_mask_zeros_padding_and_keeps_gradients() -> None:
    delta = torch.ones(1, 3, 4, 4, requires_grad=True)
    mask = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
    mask[:, :, 1:3, :] = True
    masked = apply_perturbation_mask(delta, mask)
    assert masked[:, :, (0, 3), :].count_nonzero() == 0
    assert masked[:, :, 1:3, :].eq(1).all()
    assert masked.requires_grad


def test_aligned_box_iou_identity() -> None:
    boxes = torch.tensor([[0.0, 0.0, 4.0, 4.0], [1.0, 2.0, 3.0, 8.0]])
    assert torch.allclose(box_iou_aligned(boxes, boxes), torch.ones(2))


def test_mask_generation_shape_and_iou() -> None:
    prototypes = torch.zeros(2, 4, 4)
    coefficients = torch.ones(1, 2)
    boxes = torch.tensor([[0.0, 0.0, 8.0, 8.0]])
    masks = instance_masks(coefficients, prototypes, boxes, (8, 8))
    assert masks.shape == (1, 8, 8)
    assert torch.allclose(soft_iou(masks, masks), torch.tensor([1 / 3]), atol=1e-6)
    assert torch.equal(hard_iou(masks, masks), torch.tensor([0.0]))
