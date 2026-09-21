from pathlib import Path

from yolo_mask_attack.config import load_config, require_keys


def test_base_config_has_required_sections() -> None:
    config = load_config(Path(__file__).parents[1] / "configs" / "base.yaml")
    require_keys(config, "experiment", "model", "attack", "constraints", "lagrangian")
    assert 0 < config["attack"]["epsilon"] <= 8 / 255 + 1e-12
