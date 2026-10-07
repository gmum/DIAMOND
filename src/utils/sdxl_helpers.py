import torch


def encode_prompt(pipeline, prompt, negative_prompt, device):
    prompt_embeds, neg_prompt_embeds, pooled, neg_pooled = pipeline.encode_prompt(
        prompt=prompt,
        negative_prompt=negative_prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
    )
    return prompt_embeds, neg_prompt_embeds, pooled, neg_pooled


def prepare_latents(pipeline, height, width, device, dtype, generator):
    return pipeline.prepare_latents(
        batch_size=1,
        num_channels_latents=pipeline.unet.config.in_channels,
        height=height,
        width=width,
        dtype=dtype,
        device=device,
        generator=generator,
    )


def prepare_added_conditions(
    pipeline,
    prompt_embeds,
    neg_prompt_embeds,
    pooled,
    neg_pooled,
    height,
    width,
    device,
):
    add_text_embeds = torch.cat([neg_pooled, pooled], dim=0)
    add_time_ids = pipeline._get_add_time_ids(
        original_size=(height, width),
        crops_coords_top_left=(0, 0),
        target_size=(height, width),
        dtype=prompt_embeds.dtype,
        text_encoder_projection_dim=pipeline.text_encoder_2.config.projection_dim,
    )
    add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0).to(device)
    prompt_embeds = torch.cat([neg_prompt_embeds, prompt_embeds], dim=0)
    return prompt_embeds, add_text_embeds, add_time_ids


def predict_eps(
    pipeline,
    latents,
    timestep,
    prompt_embeds,
    add_text_embeds,
    add_time_ids,
    guidance_scale,
):
    model_in = torch.cat([latents, latents])
    model_in = pipeline.scheduler.scale_model_input(model_in, timestep)
    noise_pred = pipeline.unet(
        model_in,
        timestep,
        encoder_hidden_states=prompt_embeds,
        added_cond_kwargs={
            "text_embeds": add_text_embeds,
            "time_ids": add_time_ids,
        },
        return_dict=False,
    )[0]
    n_uncond, n_text = noise_pred.chunk(2)
    return n_uncond + guidance_scale * (n_text - n_uncond)


def predict_x0_from_sigma(scheduler, latents, eps, timestep):
    if scheduler.step_index is None:
        scheduler._init_step_index(timestep)
    sigma = scheduler.sigmas[scheduler.step_index].to(
        device=latents.device,
        dtype=torch.float32,
    )
    return latents - sigma * eps.float()


def decode_latents(pipeline, latents):
    latents = latents.to(dtype=pipeline.vae.dtype)
    latents = latents / pipeline.vae.config.scaling_factor
    return pipeline.vae.decode(latents, return_dict=False)[0]
