from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import clip_boxes

from yolo_mask_attack.models.decode import decode_head
from yolo_mask_attack.models.wrapper import YoloSegWrapper

ROOT = Path(__file__).parents[1]
WEIGHTS = ROOT / "weights" / "yolov8n-seg.pt"


@pytest.mark.skipif(not WEIGHTS.is_file(), reason="official YOLOv8-Seg weight is not available")
def test_raw_decode_and_nms_match_ultralytics() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    wrapper = YoloSegWrapper(WEIGHTS, device=device)
    generator = torch.Generator(device=device).manual_seed(20260921)
    image = torch.rand((1, 3, 640, 640), generator=generator, device=device)

    with torch.no_grad():
        raw = wrapper(image)
        decoded = decode_head(raw, wrapper.strides, wrapper.reg_max)
        prediction = decoded.as_ultralytics_prediction()

    torch.testing.assert_close(prediction, raw.official_decoded, rtol=1e-5, atol=1e-5)

    ours = non_max_suppression(
        prediction.clone(), conf_thres=0.001, iou_thres=0.7, nc=wrapper.num_classes
    )[0]
    official = non_max_suppression(
        raw.official_decoded.clone(), conf_thres=0.001, iou_thres=0.7, nc=wrapper.num_classes
    )[0]
    torch.testing.assert_close(ours, official, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not WEIGHTS.is_file(), reason="official YOLOv8-Seg weight is not available")
def test_complete_predict_boxes_match_raw_pipeline() -> None:
    cv2 = pytest.importorskip("cv2")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    wrapper = YoloSegWrapper(WEIGHTS, device=device)
    image_path = ROOT / "repos" / "models" / "ultralytics" / "ultralytics" / "assets" / "zidane.jpg"
    bgr = cv2.resize(cv2.imread(str(image_path)), (640, 640))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255

    with torch.no_grad():
        raw = wrapper(image)
        decoded = decode_head(raw, wrapper.strides, wrapper.reg_max)
        detections = non_max_suppression(
            decoded.as_ultralytics_prediction().clone(),
            conf_thres=0.25,
            iou_thres=0.7,
            nc=wrapper.num_classes,
        )[0]
        clip_boxes(detections[:, :4], image.shape[-2:])
        official = wrapper.yolo.predict(
            image, conf=0.25, iou=0.7, device=device, verbose=False
        )[0].boxes.data

    torch.testing.assert_close(detections[:, :6], official, rtol=1e-4, atol=1e-4)
