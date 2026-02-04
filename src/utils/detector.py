import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation


class ArtifactDetector:
    def __init__(self, cfg, device):
        self.device = device
        self.model = SegformerForSemanticSegmentation.from_pretrained(cfg.model_id)
        self.model.decode_head.classifier = nn.Conv2d(
            self.model.decode_head.classifier.in_channels,
            1,
            1,
        )
        self.model.load_state_dict(torch.load(cfg.checkpoint_path, map_location=device))
        self.model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.input_size = tuple(cfg.input_size)
        self.mean = torch.tensor(cfg.mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(cfg.std, device=device).view(1, 3, 1, 1)

    def _normalize(self, img_01: torch.Tensor) -> torch.Tensor:
        img_resized = F.interpolate(
            img_01,
            size=self.input_size,
            mode="bilinear",
            align_corners=False,
        )
        return (img_resized - self.mean.float()) / self.std.float()

    def predict_mask(self, img: torch.Tensor) -> torch.Tensor:
        img_01 = ((img + 1) / 2).clamp(0, 1).float()
        img_norm = self._normalize(img_01)
        pred = self.model(img_norm)
        mask = torch.sigmoid(pred.logits)
        mask = F.interpolate(mask, size=img.shape[-2:], mode="bilinear", align_corners=False)
        return mask

    def max_confidence(self, img: torch.Tensor) -> float:
        with torch.no_grad():
            return self.predict_mask(img).max().item()

    def overlay_from_uint8(self, img_uint8_rgb: np.ndarray) -> np.ndarray:
        h, w, _ = img_uint8_rgb.shape

        images = torch.from_numpy(img_uint8_rgb).float() / 255.0
        images = images.permute(2, 0, 1).unsqueeze(0).to(self.device)
        images = F.interpolate(images, size=self.input_size, mode="bilinear", align_corners=False)
        images = (images - self.mean) / self.std

        with torch.no_grad():
            mask = torch.sigmoid(self.model(images).logits)

        mask = F.interpolate(mask, size=(h, w), mode="bilinear", align_corners=False)
        mask_np = mask[0, 0].cpu().numpy()

        heat_bgr = cv2.applyColorMap((mask_np * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
        overlay_rgb = cv2.addWeighted(heat_rgb, 0.6, img_uint8_rgb, 0.4, 0)
        return cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)

    def predict_mask_from_uint8(self, img_uint8_rgb: np.ndarray) -> np.ndarray:
        h, w, _ = img_uint8_rgb.shape

        images = torch.from_numpy(img_uint8_rgb).float() / 255.0
        images = images.permute(2, 0, 1).unsqueeze(0).to(self.device)
        images = F.interpolate(images, size=self.input_size, mode="bilinear", align_corners=False)
        images = (images - self.mean) / self.std

        with torch.no_grad():
            mask = torch.sigmoid(self.model(images).logits)

        mask = F.interpolate(mask, size=(h, w), mode="bilinear", align_corners=False)
        return mask[0, 0].cpu().numpy()