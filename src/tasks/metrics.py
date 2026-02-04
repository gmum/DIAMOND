import csv
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import SegformerForSemanticSegmentation
from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer


def load_prompts(path):
    prompts = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            prompts.append(row["prompt"])
    return prompts


def list_images(path):
    for fname in sorted(os.listdir(path)):
        if fname.endswith(".png"):
            yield os.path.join(path, fname)


def load_images_from_dir(path):
    images = []
    names = []
    for full_path in list_images(path):
        img = Image.open(full_path).convert("RGB")
        images.append(transforms.ToTensor()(img))  # [0,1]
        names.append(os.path.basename(full_path))
    return images, names


def load_artifact_detector(detector_cfg, device):
    model = SegformerForSemanticSegmentation.from_pretrained(detector_cfg.model_id)
    model.decode_head.classifier = nn.Conv2d(
        model.decode_head.classifier.in_channels, 1, 1
    )
    model.load_state_dict(torch.load(detector_cfg.checkpoint_path, map_location=device))
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_clip(device):
    model_name = "ViT-L-14"
    model, preprocess, _ = create_model_and_transforms(
        model_name,
        "laion2b_s32b_b82k",
        precision="fp32",
        device=device,
        output_dict=True,
    )
    tokenizer = get_tokenizer(model_name)
    model.eval()
    return model, tokenizer, preprocess


def manual_preprocess(img, mean, std):
    img = F.interpolate(img, size=(512, 512), mode="bilinear", align_corners=False)
    img = (img - mean) / std
    return img


@torch.no_grad()
def clip_t_score(images, prompts, model, tokenizer, preprocess, device, batch_size):
    scores = []
    for start in range(0, len(images), batch_size):
        end = start + batch_size
        img_batch = images[start:end]
        text_batch = prompts[start:end]
        imgs = torch.stack(
            [preprocess(transforms.ToPILImage()(im)) for im in img_batch]
        ).to(device)
        text = tokenizer(text_batch).to(device)
        out = model(imgs, text)
        img_f = F.normalize(out["image_features"], dim=-1)
        txt_f = F.normalize(out["text_features"], dim=-1)
        scores.append((img_f * txt_f).sum(dim=1) * 100.0)
    return torch.cat(scores)


def artifact_pixel_ratio(probs, thr=0.5):
    return (probs >= thr).float().mean()


@torch.no_grad()
def artifact_metrics(images, model, threshold, mean, std, device):
    image_flags = []
    pixel_ratios = []

    for img in images:
        img = img.unsqueeze(0).to(device)
        img_n = manual_preprocess(img, mean, std)

        logits = model(img_n).logits
        probs = torch.sigmoid(logits)

        image_flags.append(float(probs.max() >= threshold))
        pixel_ratios.append(artifact_pixel_ratio(probs, threshold))

    return (
        torch.tensor(image_flags),
        torch.stack(pixel_ratios),
    )


def l1_between_sets(gen_images, ref_images):
    diffs = []
    for g, r in zip(gen_images, ref_images):
        diffs.append(F.l1_loss(g, r))
    return torch.stack(diffs)


@torch.no_grad()
def artifact_mask_from_reference(ref_img, model, threshold, mean, std, device):
    img = ref_img.unsqueeze(0).to(device)
    img_n = manual_preprocess(img, mean, std)
    logits = model(img_n).logits
    probs = torch.sigmoid(logits)
    probs = F.interpolate(probs, size=ref_img.shape[-2:], mode="bilinear", align_corners=False)
    return (probs >= threshold).float()


@torch.no_grad()
def mae_by_artifact_regions(gen_images, ref_images, model, threshold, mean, std, device):
    mae_artifact = []
    mae_non_artifact = []

    for g, r in zip(gen_images, ref_images):
        mask = artifact_mask_from_reference(r, model, threshold, mean, std, device)
        mask_hw = mask[0, 0]

        diff = torch.abs(g.to(device) - r.to(device)) *255
        mask_3 = mask_hw.unsqueeze(0).expand_as(diff)
        inv_mask_3 = 1.0 - mask_3

        artifact_sum = (diff * mask_3).sum()
        artifact_count = mask_3.sum().clamp_min(1.0)
        non_sum = (diff * inv_mask_3).sum()
        non_count = inv_mask_3.sum().clamp_min(1.0)

        mae_artifact.append(artifact_sum / artifact_count)
        mae_non_artifact.append(non_sum / non_count)

    return torch.stack(mae_artifact), torch.stack(mae_non_artifact)


def run(cfg):
    device = torch.device(cfg.device)
    mean = torch.tensor(cfg.detector.mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(cfg.detector.std, device=device).view(1, 3, 1, 1)

    if not cfg.metrics.generated_dir:
        raise ValueError("metrics.generated_dir must be provided.")

    gen_images, _ = load_images_from_dir(cfg.metrics.generated_dir)
    ref_images = []
    if cfg.metrics.reference_dir:
        ref_images, _ = load_images_from_dir(cfg.metrics.reference_dir)

    prompts = []
    if cfg.metrics.prompts_csv:
        prompts = load_prompts(cfg.metrics.prompts_csv)

    if prompts and len(prompts) != len(gen_images):
        print(
            f"Warning: prompts ({len(prompts)}) and images ({len(gen_images)}) counts differ. "
            "Using the minimum length."
        )
    if ref_images and len(ref_images) != len(gen_images):
        print(
            f"Warning: reference ({len(ref_images)}) and images ({len(gen_images)}) counts differ. "
            "Using the minimum length."
        )

    min_len = len(gen_images)
    if prompts:
        min_len = min(min_len, len(prompts))
    if ref_images:
        min_len = min(min_len, len(ref_images))
    gen_images = gen_images[:min_len]
    if prompts:
        prompts = prompts[:min_len]
    if ref_images:
        ref_images = ref_images[:min_len]

    print("Generated images:", len(gen_images))
    if ref_images:
        print("Reference images:", len(ref_images))
    if prompts:
        print("Prompt[0]:", prompts[0])

    artifact_model = load_artifact_detector(cfg.detector, device)
    clip_model, clip_tokenizer, clip_preprocess = load_clip(device)

    clip_scores = None
    if prompts:
        clip_scores = clip_t_score(
            gen_images,
            prompts,
            clip_model,
            clip_tokenizer,
            clip_preprocess,
            device,
            cfg.metrics.batch_size,
        )

    artifact_flags, pixel_ratios = artifact_metrics(
        gen_images,
        artifact_model,
        cfg.metrics.artifact_threshold,
        mean,
        std,
        device,
    )

    image_l1 = None
    mae_artifact = None
    mae_non_artifact = None
    if ref_images:
        image_l1 = l1_between_sets(gen_images, ref_images) *255
        mae_artifact, mae_non_artifact = mae_by_artifact_regions(
            gen_images,
            ref_images,
            artifact_model,
            cfg.metrics.artifact_threshold,
            mean,
            std,
            device,
        )

    results_lines = ["", "===== FINAL RESULTS ====="]
    if clip_scores is not None:
        results_lines.append(f"CLIP-T                   : {clip_scores.mean():.3f}")
    results_lines.append(f"Artifact Frequency       : {artifact_flags.mean() * 100:.2f}%")
    results_lines.append(f"Artifact Pixel Ratio     : {pixel_ratios.mean() * 100:.4f}%")
    if image_l1 is not None:
        results_lines.append(f"Image L1 vs Reference    : {image_l1.mean():.6f}")
        results_lines.append(f"MAE Artifact Regions     : {mae_artifact.mean():.6f}")
        results_lines.append(f"MAE Non-Artifact Regions : {mae_non_artifact.mean():.6f}")

    print("\n".join(results_lines))

    output_dir = cfg.metrics.output_dir
    output_filename = cfg.metrics.output_filename
    if output_dir and output_filename:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(results_lines).lstrip() + "\n")
