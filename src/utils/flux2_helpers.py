from typing import List, Optional, Union

import numpy as np
import torch


def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666
    if image_seq_len > 4300:
        return float(a2 * image_seq_len + b2)
    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    return float(a * num_steps + b)


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed")
    if timesteps is not None:
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def encode_prompt(pipeline, prompt, device):
    return pipeline.encode_prompt(prompt=prompt, device=device)


def prepare_latents(pipeline, seed, height, width, device, dtype):
    generator = torch.Generator(device=device).manual_seed(seed)
    num_channels = pipeline.transformer.config.in_channels // 4
    latent, latent_ids = pipeline.prepare_latents(
        batch_size=1,
        num_latents_channels=num_channels,
        height=height,
        width=width,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    return latent, latent_ids


def predict_transformer(
    pipeline,
    latent,
    timestep,
    prompt_embeds,
    text_ids,
    latent_ids,
    device,
    dtype,
    guidance_scale: float,
):
    guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
    guidance = guidance.expand(latent.shape[0])

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
        noise_pred = pipeline.transformer(
            hidden_states=latent,
            timestep=timestep,
            guidance=guidance,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_ids,
            return_dict=False,
        )[0]

    return noise_pred[:, : latent.size(1) :]


def decode_latents(pipeline, latents, latent_ids):
    latents = pipeline._unpack_latents_with_ids(latents, latent_ids)
    bn_mean = pipeline.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
    bn_std = torch.sqrt(
        pipeline.vae.bn.running_var.view(1, -1, 1, 1) + pipeline.vae.config.batch_norm_eps
    ).to(latents.device, latents.dtype)
    latents = latents * bn_std + bn_mean
    latents = pipeline._unpatchify_latents(latents)
    return pipeline.vae.decode(latents, return_dict=False)[0]


def make_timesteps(pipeline, latents, num_steps, device):
    sigmas = np.linspace(1.0, 1.0 / num_steps, num_steps)
    if getattr(pipeline.scheduler.config, "use_flow_sigmas", False):
        sigmas = None
    mu = compute_empirical_mu(latents.shape[1], num_steps)
    timesteps, _ = retrieve_timesteps(
        pipeline.scheduler,
        num_steps,
        device,
        sigmas=sigmas,
        mu=mu,
    )
    return timesteps


def artifact_loss(mask, loss_cfg):
    loss_type = loss_cfg.type
    mask = mask.clamp(0, 1)

    if loss_type == "mse":
        return (mask**2).mean()

    if loss_type == "power":
        gamma = float(getattr(loss_cfg, "gamma", 2.0))
        return (mask**gamma).mean()

    if loss_type == "focal":
        gamma = float(getattr(loss_cfg, "gamma", 2.0))
        alpha = float(getattr(loss_cfg, "alpha", 1.0))
        eps = 1e-6
        return (-alpha * (1 - mask) ** gamma * torch.log(1 - mask + eps)).mean()

    raise ValueError(f"Unknown loss type: {loss_type}")


def compute_guidance_loss(pipeline, detector, x0_latent, latent_ids, loss_cfg):
    img = decode_latents(pipeline, x0_latent, latent_ids)
    mask = detector.predict_mask(img)
    return artifact_loss(mask, loss_cfg)
