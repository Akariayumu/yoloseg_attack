import pytest

from yolo_mask_attack.eval.reference_set import ReferenceSet, freeze_reference_set


def test_frozen_reference_set_cannot_be_overwritten(tmp_path) -> None:
    reference = ReferenceSet("v1", "model", "data", {}, ())
    destination = tmp_path / "reference.json"
    freeze_reference_set(reference, destination)
    with pytest.raises(FileExistsError):
        freeze_reference_set(reference, destination)
