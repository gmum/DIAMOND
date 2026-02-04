import numpy as np
import torch


def tensor_to_uint8(img: torch.Tensor) -> np.ndarray:
    if img.ndim == 4:
        img = img[0]
    img_np = img.detach().cpu().float().permute(1, 2, 0).numpy()
    img_np = ((img_np + 1.0) / 2.0).clip(0.0, 1.0)
    return (img_np * 255.0).astype(np.uint8)
