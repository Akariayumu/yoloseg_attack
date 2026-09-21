from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from ultralytics.utils import ops
from ultralytics.utils.nms import non_max_suppression

from yolo_mask_attack.models.decode import decode_head
from yolo_mask_attack.models.masks import instance_masks
from yolo_mask_attack.models.wrapper import YoloSegWrapper

ROOT = Path(__file__).parents[1]
WEIGHTS = ROOT / "weights" / "yolov8n-seg.pt"


@pytest.mark.skipif(not WEIGHTS.is_file(), reason="official YOLOv8-Seg weight is not available")
def test_masks_match_ultralytics_process_mask() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    wrapper = YoloSegWrapper(WEIGHTS, device=device)
    image = torch.zeros((1, 3, 640, 640), device=device)
    image[:, :, 120:520, 180:460] = 0.75

    with torch.no_grad():
        raw = wrapper(image)
        decoded = decode_head(raw, wrapper.strides, wrapper.reg_max)
        detections = non_max_suppression(
            decoded.as_ultralytics_prediction().clone(),
            conf_thres=0.0001,
            iou_thres=0.7,
            max_det=20,
            nc=wrapper.num_classes,
        )[0]

    assert len(detections) > 0
    boxes, coefficients = detections[:, :4], detections[:, 6:]
    official = ops.process_mask(
        raw.prototypes[0], coefficients, boxes.clone(), image.shape[-2:], upsample=True
    )
    ours_soft = instance_masks(
        coefficients, raw.prototypes[0], boxes, image.shape[-2:], upsample=True
    )
    ours = (ours_soft > 0.5).to(torch.uint8)
    assert torch.equal(ours, official)
