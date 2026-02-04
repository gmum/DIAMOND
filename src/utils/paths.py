import os
from typing import Optional, Tuple


def resolve_run_dir(
    base_root: str,
    model_name: str,
    run_name: str = "",
    lora_enabled: bool = False,
    dataset_name: Optional[str] = None,
) -> str:
    parts = [base_root, model_name]
    if lora_enabled:
        parts.append("lora")
    if dataset_name:
        parts.append(dataset_name)
    if run_name:
        parts.append(run_name)
    return os.path.join(*parts)


def prepare_image_dirs(root_dir: str) -> Tuple[str, str]:
    images_dir = os.path.join(root_dir, "images")
    masks_dir = os.path.join(root_dir, "masks")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    return images_dir, masks_dir


def prepare_step_dirs(root_dir: str) -> Tuple[str, str, str]:
    xt_dir = os.path.join(root_dir, "xt")
    x0_dir = os.path.join(root_dir, "x0")
    overlays_dir = os.path.join(root_dir, "overlays")
    os.makedirs(xt_dir, exist_ok=True)
    os.makedirs(x0_dir, exist_ok=True)
    os.makedirs(overlays_dir, exist_ok=True)
    return xt_dir, x0_dir, overlays_dir
