#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import torch
from ultralytics.data.augment import LetterBox

from yolo_mask_attack.attack.base import (
    AttackConfig,
    apply_perturbation_mask,
    gradient_step,
    initialize_delta,
)
from yolo_mask_attack.attack.constraints import ConstraintThresholds, constraint_values
from yolo_mask_attack.attack.lagrangian import AugmentedLagrangian, LagrangianConfig
from yolo_mask_attack.attack.objectives import (
    DynamicWeights,
    Method,
    degradation_objective,
    fixed_weight_objective,
    joint_attack_objective,
)
from yolo_mask_attack.attack.proxy import (
    ProxyObservation,
    locate_frozen_candidate,
    observe_proxy,
    proxy_metrics,
)
from yolo_mask_attack.config import load_config
from yolo_mask_attack.eval.match import classify_outcome
from yolo_mask_attack.models.decode import decode_head, pairwise_box_iou
from yolo_mask_attack.models.masks import hard_iou
from yolo_mask_attack.models.wrapper import YoloSegWrapper
from yolo_mask_attack.utils.seed import seed_everything


def prepare_image(path: Path, image_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    original = cv2.imread(str(path))
    if original is None:
        raise ValueError(f"Could not read image: {path}")
    resized = LetterBox(new_shape=(image_size, image_size), auto=False, stride=32)(image=original)
    image = (
        torch.from_numpy(resized)
        .to(device)
        .permute(2, 0, 1)
        .flip(0)
        .contiguous()
        .float()
        .div(255)
        .unsqueeze(0)
    )
    height, width = original.shape[:2]
    scale = min(image_size / height, image_size / width)
    resized_width, resized_height = round(width * scale), round(height * scale)
    left = round((image_size - resized_width) / 2 - 0.1)
    top = round((image_size - resized_height) / 2 - 0.1)
    valid_mask = torch.zeros((1, 1, image_size, image_size), dtype=torch.bool, device=device)
    valid_mask[:, :, top : top + resized_height, left : left + resized_width] = True
    return image, valid_mask


def scalar_metrics(metrics: dict[str, torch.Tensor]) -> dict[str, float]:
    return {name: float(value.detach().mean().item()) for name, value in metrics.items()}


def optimize(
    wrapper: YoloSegWrapper,
    image: torch.Tensor,
    clean: ProxyObservation,
    frozen_index: int,
    target_class: int,
    method: Method,
    attack_config: AttackConfig,
    proxy_radius: int,
    joint_config: dict[str, float],
    fixed_weights: dict[str, float],
    dynamic_config: dict[str, float],
    constrained_config: dict[str, float],
    thresholds: ConstraintThresholds,
    perturbation_mask: torch.Tensor,
) -> tuple[torch.Tensor, ProxyObservation, float]:
    delta = initialize_delta(image, attack_config.epsilon, attack_config.random_start)
    delta = apply_perturbation_mask(delta, perturbation_mask)
    dynamic_weights = DynamicWeights(
        list(fixed_weights),
        initial=float(dynamic_config["initial"]),
        growth=float(dynamic_config["growth"]),
        maximum=float(dynamic_config["maximum"]),
    )
    lagrangian = AugmentedLagrangian(
        LagrangianConfig(
            rho_initial=float(constrained_config["rho_initial"]),
            rho_growth=float(constrained_config["rho_growth"]),
            rho_max=float(constrained_config["rho_max"]),
        )
    )
    started = time.perf_counter()
    for step in range(attack_config.steps):
        raw = wrapper(image + delta)
        decoded = decode_head(raw, wrapper.strides, wrapper.reg_max)
        adversarial = observe_proxy(
            raw,
            decoded,
            frozen_index=frozen_index,
            target_class=target_class,
            input_hw=tuple(image.shape[-2:]),
            radius=proxy_radius,
        )
        metrics = proxy_metrics(clean, adversarial)
        if method is Method.MASK_ONLY:
            loss = degradation_objective(metrics["mask_iou"])
        elif method is Method.JOINT:
            loss = joint_attack_objective(
                metrics["mask_iou"],
                metrics["confidence"],
                metrics["box_iou"],
                confidence_weight=float(joint_config["confidence_weight"]),
                box_weight=float(joint_config["box_weight"]),
            )
        elif method in (Method.FIXED_WEIGHT, Method.DYNAMIC_WEIGHT, Method.CONSTRAINED):
            constraints = constraint_values(
                clean_confidence=clean.confidence,
                adversarial_confidence=adversarial.confidence,
                clean_boxes=clean.box,
                adversarial_boxes=adversarial.box,
                clean_masks=clean.mask,
                adversarial_masks=adversarial.mask,
                adversarial_logits=adversarial.logits,
                target_classes=torch.tensor([target_class], device=image.device),
                thresholds=thresholds,
            )
            if method is Method.FIXED_WEIGHT:
                loss = fixed_weight_objective(metrics["mask_iou"], constraints, fixed_weights)
            elif method is Method.DYNAMIC_WEIGHT:
                loss = fixed_weight_objective(
                    metrics["mask_iou"], constraints, dynamic_weights.values
                )
                dynamic_weights.update(constraints)
            else:
                loss = lagrangian.loss(degradation_objective(metrics["mask_iou"]), constraints)
                if (step + 1) % int(constrained_config["update_interval"]) == 0:
                    lagrangian.update(constraints)
        else:
            raise ValueError(f"Smoke runner does not implement method: {method.value}")
        delta = gradient_step(delta, image, loss, attack_config)
        delta = apply_perturbation_mask(delta, perturbation_mask)

    adversarial_image = (image + delta).detach()
    with torch.no_grad():
        raw = wrapper(adversarial_image)
        decoded = decode_head(raw, wrapper.strides, wrapper.reg_max)
        final = observe_proxy(
            raw,
            decoded,
            frozen_index=frozen_index,
            target_class=target_class,
            input_hw=tuple(image.shape[-2:]),
            radius=proxy_radius,
        )
    return adversarial_image, final, time.perf_counter() - started


def official_pair_metrics(
    wrapper: YoloSegWrapper,
    clean_image: torch.Tensor,
    adversarial_image: torch.Tensor,
    clean_proxy_box: torch.Tensor,
    target_class: int,
    base: dict[str, object],
    protocol: dict[str, object],
) -> dict[str, object]:
    predictions = wrapper.predict(
        torch.cat((clean_image, adversarial_image), dim=0),
        conf=float(base["model"]["conf_threshold"]),
        iou=float(base["model"]["iou_threshold"]),
        imgsz=int(base["model"]["image_size"]),
        retina_masks=True,
        verbose=False,
    )
    clean_result, adversarial_result = predictions
    clean_classes = clean_result.boxes.cls.long()
    eligible = (clean_classes == target_class).nonzero(as_tuple=False).squeeze(1)
    if len(eligible) == 0:
        raise RuntimeError("Frozen clean proxy did not survive official inference")
    overlaps = pairwise_box_iou(clean_proxy_box, clean_result.boxes.xyxy[eligible])[0]
    clean_index = eligible[overlaps.argmax()]
    clean_box = clean_result.boxes.xyxy[clean_index].reshape(1, 4)
    clean_confidence = clean_result.boxes.conf[clean_index]
    clean_mask = clean_result.masks.data[clean_index].reshape(1, *clean_result.masks.data.shape[-2:])

    if len(adversarial_result.boxes) == 0:
        return {"matched": False, "outcome": "missed"}
    adversarial_overlaps = pairwise_box_iou(clean_box, adversarial_result.boxes.xyxy)[0]
    adversarial_index = adversarial_overlaps.argmax()
    box_iou = float(adversarial_overlaps[adversarial_index].item())
    adversarial_class = int(adversarial_result.boxes.cls[adversarial_index].item())
    adversarial_confidence = float(adversarial_result.boxes.conf[adversarial_index].item())
    adversarial_mask = adversarial_result.masks.data[adversarial_index].reshape(
        1, *adversarial_result.masks.data.shape[-2:]
    )
    mask_iou = float(hard_iou(clean_mask, adversarial_mask).item())
    evaluation = protocol["evaluation"]
    confidence_preserved = adversarial_confidence >= (
        float(evaluation["confidence_ratio"]) * float(clean_confidence.item())
    )
    outcome = classify_outcome(
        matched=box_iou >= 0.5,
        class_preserved=adversarial_class == target_class,
        confidence_preserved=confidence_preserved,
        box_iou=box_iou,
        mask_iou=mask_iou,
        box_iou_min=float(evaluation["box_iou_min"]),
        mask_iou_max=float(evaluation["mask_iou_max"]),
    )
    return {
        "matched": box_iou >= 0.5,
        "outcome": outcome.value,
        "clean_confidence": float(clean_confidence.item()),
        "adversarial_confidence": adversarial_confidence,
        "class_preserved": adversarial_class == target_class,
        "confidence_preserved": confidence_preserved,
        "box_iou": box_iou,
        "mask_iou": mask_iou,
    }


def save_image(image: torch.Tensor, path: Path) -> None:
    rgb = image[0].mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"Could not save image: {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--protocol-config", type=Path, default=Path("configs/stage1_protocol.yaml"))
    parser.add_argument("--methods-config", type=Path, default=Path("configs/stage2_methods.yaml"))
    parser.add_argument("--reference-set", type=Path, default=Path("outputs/reference_set_v1.json"))
    parser.add_argument("--reference-index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/stage2_smoke"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int)
    args = parser.parse_args()

    base = load_config(args.base_config)
    protocol = load_config(args.protocol_config)
    methods_config = load_config(args.methods_config)
    reference_payload = json.loads(args.reference_set.read_text(encoding="utf-8"))
    reference = reference_payload["instances"][args.reference_index]
    seed = int(base["experiment"]["seed"])
    seed_everything(seed)
    image_path = Path(protocol["dataset"]["images"]) / f"{int(reference['image_id']):012d}.jpg"
    image, perturbation_mask = prepare_image(
        image_path, int(base["model"]["image_size"]), args.device
    )
    wrapper = YoloSegWrapper(base["model"]["weights"], device=args.device)

    with torch.no_grad():
        clean_raw = wrapper(image)
        clean_decoded = decode_head(clean_raw, wrapper.strides, wrapper.reg_max)
        frozen_index = locate_frozen_candidate(
            clean_decoded, int(reference["proxy_level"]), tuple(reference["proxy_yx"])
        )
        clean = observe_proxy(
            clean_raw,
            clean_decoded,
            frozen_index=frozen_index,
            target_class=int(reference["class_id"]),
            input_hw=tuple(image.shape[-2:]),
            radius=0,
        )

    smoke = methods_config["smoke"]
    attack_config = AttackConfig(
        epsilon=float(smoke["epsilon"]),
        step_size=float(smoke["step_size"]),
        steps=args.steps if args.steps is not None else int(smoke["steps"]),
        random_start=bool(smoke["random_start"]),
    )
    payload: dict[str, object] = {
        "reference_index": args.reference_index,
        "image_id": int(reference["image_id"]),
        "instance_id": int(reference["instance_id"]),
        "seed": seed,
        "attack": attack_config.__dict__,
        "valid_pixel_ratio": float(perturbation_mask.float().mean().item()),
        "sanity": {
            "zero_perturbation": official_pair_metrics(
                wrapper,
                image,
                image.clone(),
                clean.box,
                int(reference["class_id"]),
                base,
                protocol,
            )
        },
        "methods": {},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_image(image, args.output_dir / "clean.png")
    for method_name in smoke["methods"]:
        seed_everything(seed)
        method = Method(method_name)
        adversarial_image, final, elapsed = optimize(
            wrapper,
            image,
            clean,
            frozen_index,
            int(reference["class_id"]),
            method,
            attack_config,
            int(smoke["proxy_radius"]),
            methods_config["joint"],
            methods_config["fixed_weight"]["selected"],
            methods_config["dynamic_weight"],
            methods_config["constrained"],
            ConstraintThresholds(
                confidence_ratio=float(protocol["evaluation"]["confidence_ratio"]),
                box_iou_min=float(protocol["evaluation"]["box_iou_min"]),
                mask_iou_max=float(protocol["evaluation"]["mask_iou_max"]),
            ),
            perturbation_mask,
        )
        save_image(adversarial_image, args.output_dir / f"{method.value}.png")
        method_payload = {
            "elapsed_seconds": elapsed,
            "linf": float((adversarial_image - image).abs().max().item()),
            "padding_linf": float(
                ((adversarial_image - image) * ~perturbation_mask).abs().max().item()
            ),
            "proxy": scalar_metrics(proxy_metrics(clean, final)),
            "official": official_pair_metrics(
                wrapper,
                image,
                adversarial_image,
                clean.box,
                int(reference["class_id"]),
                base,
                protocol,
            ),
        }
        payload["methods"][method.value] = method_payload
        print(json.dumps({method.value: method_payload}, ensure_ascii=False, indent=2))

    output_path = args.output_dir / "metrics.json"
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"saved={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
