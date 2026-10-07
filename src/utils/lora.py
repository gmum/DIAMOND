import os
import torch


def apply_lora(pipeline, lora_cfg) -> None:
    if not lora_cfg.enabled:
        return
    if not lora_cfg.path:
        raise ValueError("LoRA is enabled but no path was provided.")

    path = str(lora_cfg.path)
    ext = os.path.splitext(path)[1].lower()
    scale = float(lora_cfg.scale)

    if ext == ".safetensors":
        pipeline.load_lora_weights(
            os.path.dirname(os.path.abspath(path)),
            weight_name=os.path.basename(path),
            adapter_name="default",
            local_files_only=True,
        )
        if scale != 1.0:
            pipeline.set_adapters("default", adapter_weights=scale)
        return

    if ext != ".bin":
        raise ValueError(f"Unsupported LoRA extension: {ext}")

    from peft import LoraConfig, set_peft_model_state_dict

    lora_config = LoraConfig(
        r=int(lora_cfg.r),
        init_lora_weights="gaussian",
        target_modules=list(lora_cfg.target_modules),
    )

    pipeline.transformer.add_adapter(lora_config, adapter_name="default")

    lora_state_dict = torch.load(path, map_location="cpu")
    set_peft_model_state_dict(
        pipeline.transformer,
        lora_state_dict,
        adapter_name="default",
    )

    pipeline.transformer.set_adapter("default")

    if scale != 1.0:
        if hasattr(pipeline.transformer, "scale_lora_layers"):
            pipeline.transformer.scale_lora_layers(scale)
