import os

import cv2
import torch

from utils.csv_utils import read_prompt_seed_csv
from utils.detector import ArtifactDetector
from utils.dtypes import resolve_dtype
from utils.images import tensor_to_uint8
from utils.lambda_schedule import build_lambda_schedule
from utils.losses import artifact_loss
from utils.lora import apply_lora
from utils.paths import prepare_image_dirs, resolve_run_dir
from utils.sdxl_helpers import (
    decode_latents,
    encode_prompt,
    predict_eps,
    prepare_added_conditions,
    prepare_latents,
)
from utils.sdxl_pipeline import build_sdxl_pipeline
from utils.seed import set_seed


def compute_guidance_loss(pipeline, detector, x0_latent, cfg):
    img = decode_latents(pipeline, x0_latent)
    mask = detector.predict_mask(img)
    return artifact_loss(mask, cfg.loss)


def run_guided(prompt, seed, pipeline, detector, cfg, device, dtype, schedule_fn):
    set_seed(seed)
    generator = torch.Generator(device=device).manual_seed(seed)

    prompt_embeds, neg_prompt_embeds, pooled, neg_pooled = encode_prompt(
        pipeline,
        prompt,
        cfg.negative_prompt,
        device,
    )
    prompt_embeds, add_text_embeds, add_time_ids = prepare_added_conditions(
        pipeline,
        prompt_embeds,
        neg_prompt_embeds,
        pooled,
        neg_pooled,
        cfg.generation.height,
        cfg.generation.width,
        device,
    )

    latents = prepare_latents(
        pipeline,
        cfg.generation.height,
        cfg.generation.width,
        device,
        prompt_embeds.dtype,
        generator,
    )

    pipeline.scheduler.set_timesteps(cfg.generation.num_steps, device=device)
    timesteps = pipeline.scheduler.timesteps
    extra_step_kwargs = pipeline.prepare_extra_step_kwargs(generator, eta=0.0)

    for step_idx, t in enumerate(timesteps):
        lambda_value = 0.0
        if cfg.guidance.enabled:
            lambda_value = schedule_fn(step_idx, cfg.generation.num_steps)

        if cfg.guidance.enabled and lambda_value != 0.0:
            latents = latents.detach().to(pipeline.unet.dtype).requires_grad_(True)
        else:
            latents = latents.detach()

        with torch.no_grad():
            model_latents = latents
            if model_latents.dtype != pipeline.unet.dtype:
                model_latents = model_latents.to(pipeline.unet.dtype)
            eps = predict_eps(
                pipeline,
                model_latents,
                t,
                prompt_embeds,
                add_text_embeds,
                add_time_ids,
                cfg.generation.guidance_scale,
            )

        step_out = pipeline.scheduler.step(
            eps,
            t,
            latents,
            return_dict=True,
            **extra_step_kwargs,
        )
        x0_latent = step_out.pred_original_sample

        shift = None
        if cfg.guidance.enabled and lambda_value != 0.0:
            loss = compute_guidance_loss(pipeline, detector, x0_latent, cfg)
            grad = torch.autograd.grad(
                loss,
                latents,
                retain_graph=False,
                create_graph=False,
            )[0]

            if cfg.guidance.normalize_grad:
                grad = grad / (grad.norm() + cfg.guidance.grad_norm_eps)

            shift = lambda_value * grad

        with torch.no_grad():
            if shift is None:
                latents = step_out.prev_sample.detach()
            else:
                latents = step_out.prev_sample.detach() - shift.to(
                    step_out.prev_sample.dtype
                )

    img = decode_latents(pipeline, latents)
    return tensor_to_uint8(img)


def run(cfg):
    device = torch.device(cfg.device)
    dtype = resolve_dtype(cfg.dtype)

    pipeline = build_sdxl_pipeline(cfg.model.model_id, device, dtype)
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
        img_uint8 = run_guided(
            prompt,
            seed,
            pipeline,
            detector,
            cfg,
            device,
            dtype,
            schedule_fn,
        )

        image_name = f"{idx:04d}_guided.png"
        cv2.imwrite(
            os.path.join(images_dir, image_name),
            cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR),
        )

        overlay = detector.overlay_from_uint8(img_uint8)
        cv2.imwrite(os.path.join(masks_dir, image_name), overlay)
