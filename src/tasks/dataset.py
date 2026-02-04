import os

import cv2
import torch

from utils.csv_utils import append_prompt_seed, ensure_csv_header, read_prompts_file
from utils.detector import ArtifactDetector
from utils.dtypes import resolve_dtype
from utils.flux_helpers import decode_latents, encode_prompt, prepare_latents, predict_transformer
from utils.flux_pipeline import build_flux_pipeline
from utils.images import tensor_to_uint8
from utils.paths import prepare_image_dirs, resolve_run_dir
from utils.scheduling import make_scheduler_and_timesteps
from utils.seed import set_seed


@torch.no_grad()
def generate_baseline_image(pipeline, prompt, seed, cfg, device, dtype):
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

    for t in timesteps:
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

        latent = scheduler.step(pred, t, latent, return_dict=False)[0]

    return decode_latents(
        pipeline,
        latent,
        cfg.generation.height,
        cfg.generation.width,
    )


def run(cfg):
    device = torch.device(cfg.device)
    dtype = resolve_dtype(cfg.dtype)

    pipeline = build_flux_pipeline(cfg.model.model_id, device, dtype)
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
