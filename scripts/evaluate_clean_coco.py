#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO

from yolo_mask_attack.config import load_config

STAT_NAMES = (
    "ap",
    "ap50",
    "ap75",
    "ap_small",
    "ap_medium",
    "ap_large",
    "ar_1",
    "ar_10",
    "ar_100",
    "ar_small",
    "ar_medium",
    "ar_large",
)


def encode_masks(masks: np.ndarray) -> list[dict[str, object]]:
    encoded = mask_utils.encode(np.asfortranarray(masks.transpose(1, 2, 0).astype(np.uint8)))
    if isinstance(encoded, dict):
        encoded = [encoded]
    return [
        {"size": list(rle["size"]), "counts": rle["counts"].decode("ascii")} for rle in encoded
    ]


def evaluate(
    ground_truth: COCO,
    predictions: list[dict[str, object]],
    iou_type: str,
    category_ids: list[int] | None = None,
) -> dict[str, float]:
    detection_results = ground_truth.loadRes(predictions)
    evaluator = COCOeval(ground_truth, detection_results, iou_type)
    evaluator.params.imgIds = sorted(ground_truth.getImgIds())
    if category_ids is not None:
        evaluator.params.catIds = category_ids
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return {name: float(value) for name, value in zip(STAT_NAMES, evaluator.stats)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--protocol-config", type=Path, default=Path("configs/stage1_protocol.yaml"))
    parser.add_argument("--output", type=Path, default=Path("outputs/clean_coco_metrics.json"))
    parser.add_argument("--predictions", type=Path, default=Path("outputs/clean_coco_predictions.json"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--rect", action="store_true")
    args = parser.parse_args()

    base = load_config(args.base_config)
    protocol = load_config(args.protocol_config)
    dataset_config = protocol["dataset"]
    ground_truth = COCO(dataset_config["annotations"])
    images = [ground_truth.imgs[image_id] for image_id in sorted(ground_truth.getImgIds())]
    image_by_file = {image["file_name"]: image for image in images}
    category_by_name = {category["name"]: category["id"] for category in ground_truth.loadCats(ground_truth.getCatIds())}

    model = YOLO(base["model"]["weights"])
    predictions: list[dict[str, object]] = []
    started = time.perf_counter()
    results = model.predict(
        source=str(dataset_config["images"]),
        stream=True,
        batch=args.batch,
        rect=args.rect,
        imgsz=int(base["model"]["image_size"]),
        conf=0.001,
        iou=float(base["model"]["iou_threshold"]),
        max_det=300,
        retina_masks=True,
        device=args.device,
        verbose=False,
    )
    for index, result in enumerate(results, start=1):
        image_info = image_by_file[Path(result.path).name]
        if result.boxes is not None and len(result.boxes):
            boxes_xywh = result.boxes.xywh.cpu().numpy()
            boxes_xywh[:, :2] -= boxes_xywh[:, 2:] / 2
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.int().cpu().tolist()
            masks = encode_masks(result.masks.data.cpu().numpy()) if result.masks is not None else []
            if len(masks) != len(classes):
                raise RuntimeError("Detection and mask counts differ")
            for box, score, class_index, mask in zip(boxes_xywh, scores, classes, masks):
                class_name = model.names[class_index]
                predictions.append(
                    {
                        "image_id": int(image_info["id"]),
                        "category_id": int(category_by_name[class_name]),
                        "bbox": [round(float(value), 3) for value in box],
                        "score": round(float(score), 6),
                        "segmentation": mask,
                    }
                )
        if index % 500 == 0 or index == len(images):
            elapsed = time.perf_counter() - started
            print(f"processed={index}/{len(images)} predictions={len(predictions)} elapsed={elapsed:.1f}s")

    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    args.predictions.write_text(json.dumps(predictions, separators=(",", ":")), encoding="utf-8")
    person_id = category_by_name["person"]
    metrics = {
        "config": {
            "weights": base["model"]["weights"],
            "model_sha256": base["model"]["sha256"],
            "ultralytics_version": base["model"]["ultralytics_version"],
            "images": len(images),
            "image_size": int(base["model"]["image_size"]),
            "confidence_threshold": 0.001,
            "nms_iou_threshold": float(base["model"]["iou_threshold"]),
            "max_detections": 300,
            "retina_masks": True,
            "batch": args.batch,
            "rectangular_inference": args.rect,
            "device": args.device,
        },
        "prediction_count": len(predictions),
        "elapsed_seconds": time.perf_counter() - started,
        "all_categories": {
            "bbox": evaluate(ground_truth, predictions, "bbox"),
            "segm": evaluate(ground_truth, predictions, "segm"),
        },
        "person": {
            "bbox": evaluate(ground_truth, predictions, "bbox", [person_id]),
            "segm": evaluate(ground_truth, predictions, "segm", [person_id]),
        },
    }
    args.output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
