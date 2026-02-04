from diffusers import FluxPipeline


def build_flux_pipeline(model_id, device, dtype):
    pipeline = FluxPipeline.from_pretrained(model_id, torch_dtype=dtype).to(device)
    pipeline.safety_checker = None
    pipeline.enable_attention_slicing()
    pipeline.vae.enable_tiling()
    pipeline.vae.enable_slicing()

    pipeline.transformer.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.vae.requires_grad_(False)
    return pipeline
