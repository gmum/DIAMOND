import os

import cv2
import torch

from utils.csv_utils import append_prompt_seed, ensure_csv_header, read_prompts_file
from utils.detector import ArtifactDetector
from utils.dtypes import resolve_dtype
from utils.images import tensor_to_uint8
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


@torch.no_grad()
def generate_baseline_image(pipeline, prompt, seed, cfg, device, dtype):
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

    for t in timesteps:
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
        latents = pipeline.scheduler.step(
            eps,
            t,
            latents,
            return_dict=False,
            **extra_step_kwargs,
        )[0]

    return decode_latents(pipeline, latents)


def run(cfg):
    device = torch.device(cfg.device)
    dtype = resolve_dtype(cfg.dtype)

    pipeline = build_sdxl_pipeline(cfg.model.model_id, device, dtype)
    detector = ArtifactDetector(cfg.detector, device)

    prompts = read_prompts_file(cfg.dataset.prompts_file)

    run_root = resolve_run_dir(
        cfg.paths.dataset_root,
        cfg.model.name,
        run_name=cfg.output.run_name,
        lora_enabled=False,
        dataset_name=cfg.dataset.name,
    )
    images_dir, masks_dir = prepare_image_dirs(run_root)

    results_csv = os.path.join(run_root, cfg.dataset.results_csv)
    ensure_csv_header(results_csv)

    global_seed = int(cfg.seed)

    for pid, prompt in enumerate(prompts):
        found = False

        for _ in range(cfg.dataset.max_tries_per_prompt):
            seed = global_seed
            global_seed += 1

            img = generate_baseline_image(pipeline, prompt, seed, cfg, device, dtype)
            score = detector.max_confidence(img)

            if score >= cfg.dataset.artifact_threshold:
                img_uint8 = tensor_to_uint8(img)
                image_name = f"prompt{pid:03d}_seed{seed}.png"

                cv2.imwrite(
                    os.path.join(images_dir, image_name),
                    cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR),
                )

                if cfg.dataset.save_overlays:
                    overlay = detector.overlay_from_uint8(img_uint8)
                    cv2.imwrite(os.path.join(masks_dir, image_name), overlay)

                append_prompt_seed(results_csv, prompt, seed)
                found = True
                break

        if not found:
            print(f"No artifact >= {cfg.dataset.artifact_threshold} for prompt {pid}")
