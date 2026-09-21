from pathlib import Path

import pytest

from yolo_mask_attack.data import CocoPersonDataset

ROOT = Path(__file__).parents[1]
IMAGES = ROOT / "data" / "coco" / "val2017"
ANNOTATIONS = ROOT / "data" / "coco" / "annotations" / "instances_val2017.json"


@pytest.mark.skipif(not ANNOTATIONS.is_file(), reason="COCO val2017 is not available")
def test_person_dataset_is_deterministic_and_masks_align() -> None:
    dataset = CocoPersonDataset(IMAGES, ANNOTATIONS)
    assert len(dataset) == 2693
    first = dataset[0]
    assert first.path.is_file()
    assert first.annotation_ids == tuple(sorted(first.annotation_ids))
    assert dataset.boxes_xyxy(first).shape == (len(first.annotation_ids), 4)
    assert dataset.masks(first).shape == (len(first.annotation_ids), first.height, first.width)
    assert [record.image_id for record in dataset.records(limit=5, seed=7)] == [
        record.image_id for record in dataset.records(limit=5, seed=7)
    ]
