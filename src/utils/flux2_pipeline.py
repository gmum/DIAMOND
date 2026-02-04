from diffusers import Flux2Pipeline


def build_flux2_pipeline(model_id, dtype):
    pipeline = Flux2Pipeline.from_pretrained(model_id, torch_dtype=dtype)

    pipeline.vae.requires_grad_(False)
    pipeline.transformer.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)

    try:
        pipeline.vae.enable_slicing()
    except Exception:
        pass
    try:
        pipeline.vae.enable_tiling()
    except Exception:
        pass
    try:
        pipeline.enable_attention_slicing()
    except Exception:
        pass

    pipeline.enable_model_cpu_offload()
    return pipeline
