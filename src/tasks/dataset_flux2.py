import os
import random

import cv2
import numpy as np
import torch

from utils.csv_utils import append_prompt_seed, ensure_csv_header, read_prompts_file
from utils.dtypes import resolve_dtype
from utils.flux2_detector import Flux2ArtifactDetector
from utils.flux2_helpers import decode_latents, encode_prompt, make_timesteps, predict_transformer, prepare_latents
from utils.flux2_pipeline import build_flux2_pipeline
from utils.images import tensor_to_uint8
from utils.paths import prepare_image_dirs, resolve_run_dir


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def generate_baseline_image(pipeline, prompt, seed, cfg, device, dtype):
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

    for t in timesteps:
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
        latent = pipeline.scheduler.step(noise_pred, t, latent, return_dict=False)[0]

    return decode_latents(pipeline, latent, latent_ids)


def run(cfg):
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    device = torch.device(cfg.device)
    dtype = resolve_dtype(cfg.dtype)

    pipeline = build_flux2_pipeline(cfg.model.model_id, dtype)
    detector = Flux2ArtifactDetector(cfg.detector, device, dtype=torch.float32)

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
