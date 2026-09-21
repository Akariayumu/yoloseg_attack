#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_utils
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops
from ultralytics.utils.nms import non_max_suppression

from yolo_mask_attack.config import load_config
from yolo_mask_attack.data import CocoPersonDataset
from yolo_mask_attack.eval.match import greedy_iou_match
from yolo_mask_attack.eval.reference_set import (
    ReferenceInstance,
    ReferenceSet,
    freeze_reference_set,
)
from yolo_mask_attack.models.decode import decode_head
from yolo_mask_attack.models.wrapper import YoloSegWrapper


def encode_mask(mask: np.ndarray) -> dict[str, object]:
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    return {"size": list(rle["size"]), "counts": rle["counts"].decode("ascii")}


def binary_iou(mask_a: torch.Tensor, mask_b: torch.Tensor) -> float:
    intersection = (mask_a & mask_b).sum()
    union = (mask_a | mask_b).sum()
    return float((intersection.float() / union.clamp_min(1)).item())


def build(args: argparse.Namespace) -> ReferenceSet:
    base = load_config(args.base_config)
    protocol = load_config(args.protocol_config)
    dataset_config = protocol["dataset"]
    reference_config = protocol["reference"]
    dataset = CocoPersonDataset(dataset_config["images"], dataset_config["annotations"])
    records = dataset.records(limit=args.limit, seed=args.seed if args.limit else None)
    wrapper = YoloSegWrapper(base["model"]["weights"], device=args.device)
    image_size = int(base["model"]["image_size"])
    letterbox = LetterBox(new_shape=(image_size, image_size), auto=False, stride=32)
    instances: list[ReferenceInstance] = []

    for position, record in enumerate(records, start=1):
        original = cv2.imread(str(record.path))
        if original is None:
            raise ValueError(f"Could not read image: {record.path}")
        resized = letterbox(image=original)
        image = (
            torch.from_numpy(resized)
            .to(args.device)
            .permute(2, 0, 1)
            .flip(0)
            .contiguous()
            .float()
            .div(255)
            .unsqueeze(0)
        )
        with torch.no_grad():
            raw = wrapper(image)
            decoded = decode_head(raw, wrapper.strides, wrapper.reg_max)
            detections, kept_indices = non_max_suppression(
                decoded.as_ultralytics_prediction().clone(),
                conf_thres=float(base["model"]["conf_threshold"]),
                iou_thres=float(base["model"]["iou_threshold"]),
                classes=[0],
                max_det=300,
                nc=wrapper.num_classes,
                return_idxs=True,
            )
        detections, kept_indices = detections[0], kept_indices[0].reshape(-1)
        if len(detections) == 0:
            continue

        input_boxes = detections[:, :4].clone()
        original_boxes = ops.scale_boxes(
            image.shape[-2:], input_boxes.clone(), original.shape[:2]
        )
        predicted_masks = ops.process_mask_native(
            raw.prototypes[0], detections[:, 6:], original_boxes, original.shape[:2]
        ).bool()

        annotations = [
            annotation
            for annotation in dataset.annotations_for(record)
            if not annotation.get("iscrowd", 0)
        ]
        if not annotations:
            continue
        gt_boxes = torch.tensor(
            [
                [
                    annotation["bbox"][0],
                    annotation["bbox"][1],
                    annotation["bbox"][0] + annotation["bbox"][2],
                    annotation["bbox"][1] + annotation["bbox"][3],
                ]
                for annotation in annotations
            ],
            device=args.device,
        )
        gt_indices, prediction_indices, box_ious = greedy_iou_match(
            gt_boxes,
            original_boxes,
            iou_threshold=float(reference_config["ground_truth_box_iou_min"]),
        )
        for gt_index, prediction_index, box_iou in zip(
            gt_indices.tolist(), prediction_indices.tolist(), box_ious.tolist()
        ):
            gt_mask = torch.from_numpy(dataset.coco.annToMask(annotations[gt_index])).to(
                args.device, dtype=torch.bool
            )
            mask_iou = binary_iou(gt_mask, predicted_masks[prediction_index])
            if mask_iou < float(reference_config["ground_truth_mask_iou_min"]):
                continue
            candidate_index = int(kept_indices[prediction_index].item())
            prediction_box = original_boxes[prediction_index].tolist()
            instances.append(
                ReferenceInstance(
                    image_id=record.image_id,
                    instance_id=int(annotations[gt_index]["id"]),
                    class_id=0,
                    confidence=float(detections[prediction_index, 4].item()),
                    box_xyxy=tuple(float(value) for value in prediction_box),
                    mask_rle=encode_mask(predicted_masks[prediction_index].cpu().numpy()),
                    proxy_level=int(decoded.levels[candidate_index].item()),
                    proxy_yx=tuple(int(value) for value in decoded.grid_yx[candidate_index].tolist()),
                    ground_truth_box_iou=float(box_iou),
                    ground_truth_mask_iou=mask_iou,
                )
            )
        if position % 20 == 0 or position == len(records):
            print(f"processed={position}/{len(records)} references={len(instances)}")

    version = protocol["version"] if args.limit is None else f"{protocol['version']}-smoke-{args.limit}"
    return ReferenceSet(
        version=version,
        model_sha256=base["model"]["sha256"],
        dataset_sha256=dataset_config["instances_val2017_sha256"],
        protocol={
            "model": base["model"],
            "reference": reference_config,
            "evaluation": protocol["evaluation"],
            "sample_limit": args.limit,
            "seed": args.seed,
        },
        instances=tuple(instances),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--protocol-config", type=Path, default=Path("configs/stage1_protocol.yaml"))
    parser.add_argument("--output", type=Path, default=Path("outputs/reference_set_smoke20.json"))
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--full", action="store_true", help="process the complete person subset")
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.full:
        args.limit = None
    reference_set = build(args)
    if not reference_set.instances:
        raise RuntimeError("No reference instances passed the clean-quality thresholds")
    freeze_reference_set(reference_set, args.output)
    print(f"frozen={args.output} instances={len(reference_set.instances)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
