import torch


def encode_prompt(pipeline, prompt, device, dtype, max_sequence_length: int = 256):
    prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
        prompt=[prompt],
        prompt_2=[prompt],
        device=device,
        num_images_per_prompt=1,
        max_sequence_length=max_sequence_length,
    )
    prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
    pooled_prompt_embeds = pooled_prompt_embeds.to(device=device, dtype=dtype)
    text_ids = text_ids.to(device=device)
    return prompt_embeds, pooled_prompt_embeds, text_ids


def prepare_latents(pipeline, seed, height, width, device, dtype):
    generator = torch.Generator(device=device).manual_seed(seed)
    num_channels = pipeline.transformer.config.in_channels // 4
    latent, latent_ids = pipeline.prepare_latents(
        batch_size=1,
        num_channels_latents=num_channels,
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
    pooled_prompt_embeds,
    prompt_embeds,
    text_ids,
    latent_ids,
    device,
    dtype,
    guidance_scale: float,
):
    if getattr(pipeline.transformer.config, "guidance_embeds", False):
        guidance = torch.full((latent.shape[0],), guidance_scale, device=device, dtype=dtype)
    else:
        guidance = None

    return pipeline.transformer(
        hidden_states=latent,
        timestep=timestep,
        guidance=guidance,
        pooled_projections=pooled_prompt_embeds,
        encoder_hidden_states=prompt_embeds,
        txt_ids=text_ids,
        img_ids=latent_ids,
        joint_attention_kwargs=None,
        return_dict=False,
    )[0]


def decode_latents(pipeline, latents, height, width):
    lv = pipeline._unpack_latents(latents, height, width, pipeline.vae_scale_factor)
    lv = (lv / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
    return pipeline.vae.decode(lv.to(device=latents.device, dtype=pipeline.vae.dtype)).sample
