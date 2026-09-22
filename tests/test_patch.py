import pytest

torch = pytest.importorskip("torch")

from yolo_mask_attack.attack.patch import rectangular_patch_mask


def test_rectangular_patch_is_inside_target_box() -> None:
    box = torch.tensor([[20.0, 10.0, 60.0, 90.0]])
    mask = rectangular_patch_mask(
        box, (100, 100), width_ratio=0.5, height_ratio=0.25, vertical_position=0.4
    )
    rows, cols = mask[0, 0].nonzero(as_tuple=True)
    assert mask.sum().item() == 20 * 20
    assert cols.min() >= 20 and cols.max() < 60
    assert rows.min() >= 10 and rows.max() < 90


def test_rectangular_patch_rejects_invalid_ratio() -> None:
    with pytest.raises(ValueError, match="ratios"):
        rectangular_patch_mask(torch.zeros(1, 4), (10, 10), width_ratio=0)
