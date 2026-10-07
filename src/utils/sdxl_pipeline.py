import torch
from diffusers import StableDiffusionXLPipeline


def build_sdxl_pipeline(
    model_id,
    device,
    dtype,
    vae_dtype=torch.float32,
):
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
    ).to(device)

    pipeline.set_progress_bar_config(disable=True)
    pipeline.enable_attention_slicing()
    pipeline.enable_vae_slicing()

    pipeline.unet.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.vae.requires_grad_(False)
    pipeline.vae.to(dtype=vae_dtype)

    return pipeline
