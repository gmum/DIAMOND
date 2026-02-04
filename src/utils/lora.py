import os
import torch


def apply_lora(pipeline, lora_cfg) -> None:
    if not lora_cfg.enabled:
        return
    if not lora_cfg.path:
        raise ValueError("LoRA is enabled but no path was provided.")

    path = str(lora_cfg.path)
    ext = os.path.splitext(path)[1].lower()

    if ext == ".safetensors":
        pipeline.load_lora_weights(path)
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