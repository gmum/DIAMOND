import os
import random

import cv2
import numpy as np
import torch

from utils.csv_utils import read_prompt_seed_csv
from utils.dtypes import resolve_dtype
from utils.flux2_detector import Flux2ArtifactDetector
from utils.flux2_helpers import compute_guidance_loss, decode_latents, encode_prompt, make_timesteps, predict_transformer, prepare_latents
from utils.flux2_pipeline import build_flux2_pipeline
from utils.images import tensor_to_uint8
from utils.lambda_schedule import build_lambda_schedule
from utils.lora import apply_lora
from utils.paths import prepare_image_dirs, resolve_run_dir


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_guided(prompt, seed, pipeline, detector, cfg, device, dtype, schedule_fn):
    set_seed(seed)

    prompt_embeds, text_ids = encode_prompt(pipeline, prompt, device)
    latent, latent_ids = prepare_latents(
        pipeline,
        seed,
        cfg.generation.height,
        cfg.generation.width,
        device,
        dtype,
    )

    timesteps = make_timesteps(
        pipeline,
        latent,
        cfg.generation.num_steps,
        device,
    )
    pipeline.scheduler.set_begin_index(0)

    for step_idx, t in enumerate(timesteps):
        if device.type == "cuda":
            torch.cuda.empty_cache()

        lambda_value = 0.0
        if cfg.guidance.enabled:
            lambda_value = schedule_fn(step_idx, cfg.generation.num_steps)

        if cfg.guidance.enabled and lambda_value != 0.0:
            latent = latent.detach().requires_grad_(True)
        else:
            latent = latent.detach()

        t_vec = t.expand(latent.shape[0]).to(latent.dtype)
        noise_pred = predict_transformer(
            pipeline,
            latent,
            t_vec / 1000.0,
            prompt_embeds,
            text_ids,
            latent_ids,
            device,
            dtype,
            cfg.generation.guidance_scale,
        )

        x0_latent = latent - (t_vec / 1000.0).view(-1, 1, 1) * noise_pred

        shift = None
        if cfg.guidance.enabled and lambda_value != 0.0:
            loss = compute_guidance_loss(pipeline, detector, x0_latent, latent_ids, cfg.loss)
            grad = torch.autograd.grad(loss, latent, retain_graph=False, create_graph=False)[0]

            if cfg.guidance.normalize_grad:
                grad = grad / (grad.norm() + cfg.guidance.grad_norm_eps)

            shift = (-lambda_value * grad).to(latent.dtype)

        with torch.no_grad():
            latent = pipeline.scheduler.step(noise_pred, t, latent, return_dict=False)[0]
            if shift is not None:
                latent = latent + shift

    img = decode_latents(pipeline, latent, latent_ids)
    return tensor_to_uint8(img)


def run(cfg):
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    device = torch.device(cfg.device)
    dtype = resolve_dtype(cfg.dtype)

    pipeline = build_flux2_pipeline(cfg.model.model_id, dtype)
    apply_lora(pipeline, cfg.lora)

    detector = Flux2ArtifactDetector(cfg.detector, device, dtype=torch.float32)
    schedule_fn = build_lambda_schedule(cfg.lambda_schedule)

    run_root = resolve_run_dir(
        cfg.paths.output_root,
        cfg.model.name,
        run_name=cfg.output.run_name,
        lora_enabled=cfg.lora.enabled,
    )
    images_dir, masks_dir = prepare_image_dirs(run_root)

    rows = read_prompt_seed_csv(cfg.csv_path)

    for idx, (prompt, seed) in enumerate(rows):
        print(f"Generating {idx + 1}/{len(rows)}")
        img_uint8 = run_guided(prompt, seed, pipeline, detector, cfg, device, dtype, schedule_fn)

        image_name = f"{idx:04d}_guided.png"
        cv2.imwrite(
            os.path.join(images_dir, image_name),
            cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR),
        )

        overlay = detector.overlay_from_uint8(img_uint8)
        cv2.imwrite(os.path.join(masks_dir, image_name), overlay)
