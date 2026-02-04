import os

import cv2
import torch

from utils.csv_utils import read_prompt_seed_csv
from utils.detector import ArtifactDetector
from utils.dtypes import resolve_dtype
from utils.flux_helpers import decode_latents, encode_prompt, prepare_latents, predict_transformer
from utils.flux_pipeline import build_flux_pipeline
from utils.images import tensor_to_uint8
from utils.lambda_schedule import build_lambda_schedule
from utils.lora import apply_lora
from utils.losses import artifact_loss
from utils.paths import prepare_image_dirs, resolve_run_dir
from utils.scheduling import make_scheduler_and_timesteps
from utils.seed import set_seed


def compute_guidance_loss(pipeline, detector, x0_latent, cfg):
    img = decode_latents(
        pipeline,
        x0_latent,
        cfg.generation.height,
        cfg.generation.width,
    )
    mask = detector.predict_mask(img)
    return artifact_loss(mask, cfg.loss)


def run_guided(prompt, seed, idx, pipeline, detector, cfg, device, dtype, schedule_fn):
    set_seed(seed)

    prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
        pipeline,
        prompt,
        device,
        dtype,
        max_sequence_length=cfg.generation.max_sequence_length,
    )

    latent, latent_ids = prepare_latents(
        pipeline,
        seed,
        cfg.generation.height,
        cfg.generation.width,
        device,
        dtype,
    )

    scheduler, timesteps = make_scheduler_and_timesteps(
        pipeline,
        latent,
        cfg.generation.num_steps,
        device,
    )

    for step_idx, t in enumerate(timesteps):
        lambda_value = 0.0
        if cfg.guidance.enabled:
            lambda_value = schedule_fn(step_idx, cfg.generation.num_steps)

        if cfg.guidance.enabled and lambda_value != 0.0:
            latent = latent.detach().requires_grad_(True)
        else:
            latent = latent.detach()

        with torch.no_grad():
            t_vec = t.expand(latent.shape[0]).to(device, dtype)
            pred = predict_transformer(
                pipeline,
                latent,
                t_vec / 1000.0,
                pooled_prompt_embeds,
                prompt_embeds,
                text_ids,
                latent_ids,
                device,
                dtype,
                cfg.generation.guidance_scale,
            )

        if cfg.guidance.enabled and lambda_value != 0.0:
            step_out = scheduler.step(pred, t, latent, return_dict=True)
            x0_latent = step_out.pred_original_sample

            loss = compute_guidance_loss(pipeline, detector, x0_latent, cfg)
            grad = torch.autograd.grad(loss, latent)[0]

            if cfg.guidance.normalize_grad:
                grad = grad / (grad.norm() + cfg.guidance.grad_norm_eps)
                #grad = grad

            shift = (-lambda_value * grad).to(latent.dtype)

            with torch.no_grad():
                latent = step_out.prev_sample.detach() + shift
        else:
            with torch.no_grad():
                step_out = scheduler.step(pred, t, latent, return_dict=True)
                latent = step_out.prev_sample.detach()

    img = decode_latents(
        pipeline,
        latent,
        cfg.generation.height,
        cfg.generation.width,
    )
    img_uint8 = tensor_to_uint8(img)
    return img_uint8


def run(cfg):
    device = torch.device(cfg.device)
    dtype = resolve_dtype(cfg.dtype)

    pipeline = build_flux_pipeline(cfg.model.model_id, device, dtype)
    apply_lora(pipeline, cfg.lora)

    detector = ArtifactDetector(cfg.detector, device)
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
        img_uint8 = run_guided(prompt, seed, idx, pipeline, detector, cfg, device, dtype, schedule_fn)

        image_name = f"{idx:04d}_guided.png"
        cv2.imwrite(
            os.path.join(images_dir, image_name),
            cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR),
        )

        overlay = detector.overlay_from_uint8(img_uint8)
        cv2.imwrite(os.path.join(masks_dir, image_name), overlay)
