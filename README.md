# DIAMOND: Directed Inference for Artifact Mitigation in Flow Matching Models

<div align="center">

🌐 **[Project Page](https://gmum.github.io/DIAMOND/)** &nbsp;&nbsp;|&nbsp;&nbsp; 📄 **[arXiv](https://arxiv.org/abs/2602.00883)**
<br>

[Alicja Polowczyk*](https://www.linkedin.com/in/alicja-polowczyk-064739266/), [Agnieszka Polowczyk*](https://www.linkedin.com/in/agnieszka-polowczyk-91381323a/), [Piotr Borycki](https://www.linkedin.com/in/piotr-borycki-560052251), [Joanna Waczyńska](https://www.linkedin.com/in/joannawaczynska/), [Jacek Tabor](https://scholar.google.pl/citations?user=zSKYziUAAAAJ&hl=pl), [Przemysław Spurek](https://scholar.google.com/citations?hl=en&user=0kp0MbgAAAAJ)  
(*equal contribution)


</div>

---

<p align="center">
<img src="assets/teaser.jpg" width="92%">
</p>

**DIAMOND** is a *training-free, inference-time guidance framework* that tackles one of the most persistent challenges in modern text-to-image generation: **visual and anatomical artifacts**.

While recent models such as FLUX achieve impressive realism, they still frequently produce distorted structures, malformed anatomy, and visual inconsistencies. Unlike existing post-hoc or weight-modifying approaches, DIAMOND intervenes **directly during the generative process** by reconstructing a clean sample estimate at each step and **steering the sampling trajectory away from artifact-prone latent states**.

The method requires **no additional training, no finetuning, and no weight modification**, and can be applied to both **flow matching models and standard diffusion models**, enabling robust, zero-shot, high-fidelity image synthesis with substantially reduced artifacts.

---

## 📰 News

- **Feb. 2026**: Initial codebase released with support for **FLUX models** (FLUX.1-dev, FLUX-schnell, FLUX-2-dev).
- **Feb. 2026**: Paper is available on arXiv.
- **Coming Soon**: **SDXL code** will be added to the repository.


## ⚙️ Environment Setup

We provide two separate environment configurations depending on the model variant.

### 🔹 Option A — FLUX.1 [dev], FLUX.1 [schnell], SDXL

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-red)
![TorchVision](https://img.shields.io/badge/torchvision-0.21.0-orange)
![Diffusers](https://img.shields.io/badge/diffusers-0.33.1-yellow)

Create and activate the Conda environment:

```bash
conda create -n diamond python=3.11 -y
conda activate diamond
```
Install PyTorch and remaining dependencies:
```bash
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

### 🔹 Option B — FLUX-2-dev
Requires a newer version of diffusers installed directly from GitHub.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red)
![TorchVision](https://img.shields.io/badge/torchvision-0.20.1-orange)
![TorchAudio](https://img.shields.io/badge/torchaudio-2.5.1-orange)
![Diffusers](https://img.shields.io/badge/diffusers-github-yellow)

```bash
conda create -n diamond-flux2 python=3.10 -y
conda activate diamond-flux2

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu118

pip uninstall diffusers -y
pip install git+https://github.com/huggingface/diffusers.git -U

pip install -r requirements2.txt

```
## 📦 SOTA Method Weights

We release **our trained model weights** for several state-of-the-art artifact mitigation methods.


| Base Model        | DiffDoctor | HPSv2 | HandsXL |
|-----------------|------------|-------|---------|
| FLUX.1 [dev]    | Coming Soon | Coming Soon | Coming Soon |
| FLUX.1 [schnell]| Coming Soon | Coming Soon | — |
| SDXL            | — | — | Coming Soon |
| FLUX.2 [dev]   | — | — | — |

Full evaluation datasets (CSV files with prompts and corresponding random seeds) are provided in the `datasets/` directory.  
For **SDXL**, a shortened dataset variant is released, as no random seeds producing artifact-containing images could be found for some prompts.

# DIAMOND

## 🚀 Generate a Single Image

Move to the repository root:

```bash
cd DIAMOND
```
You can select the base model using `model=dev` (**FLUX.1 [dev]**) or `model=schnell` (FLUX.1 **[schnell]**).
Setting `guidance.enabled=true` enables **DIAMOND guidance** during sampling. To run **without DIAMOND (baseline)**, set `guidance.enabled=false`.
You can also modify the `loss` type and the `lambda_schedule` to explore different guidance behaviors.

### Run Generation
```bash
python src/generate_single_image.py \
  model=dev \
  'prompt="Luxury crystal blue diamond, premium brand mark, vector style, simple and iconic, 4k resolution"' \
  seed=100285 \
  guidance.enabled=false \
  loss=power \
  lambda_schedule=power \
  lambda_schedule.start=25 \
  lambda_schedule.end=1 \
  lambda_schedule.power=2 \
  output.run_name=example_run
```

For **FLUX.2 [dev]**, use the separate script:
```bash
python src/generate_single_image_flux2.py \
  model=flux2dev \
  'prompt="Luxury crystal blue diamond, premium brand mark, vector style, simple and iconic, 4k resolution"' \
  seed=100285 \
  output.run_name=example_run
```
> [!IMPORTANT]
> Activate the correct Conda environment before running (see Environment Setup).
> Outputs are saved to the `outputs/` directory.

### LoRA-based SOTA Methods
See the **📦 SOTA Method Weights** table for model support. Enable LoRA and set the appropriate checkpoint in `lora.path`.

### Example (HandsXL)

```bash
python src/generate_single_image.py \
  model=dev \
  'prompt="A South Asian man, 35 years old, with a visual impairment, reading braille books in a library."' \
  seed=100283 \
  lora=enabled \
  lora.path="checkpoints/lora/people_handv1.safetensors" \
  guidance.enabled=false \
  output.run_name=lora_example
```
> [!IMPORTANT]
> When using LoRA-based SOTA methods, always set `guidance.enabled=false`.

## 🚀 Generate Multiple Images
The generation setup is identical to single-image generation. **DIAMOND** can be enabled or disabled using `guidance.enabled=true/false`.  
**LoRA-based SOTA** methods can be used by setting `lora=enabled` and specifying `lora.path`.
 
For **FLUX.1 [dev]**, **FLUX.1 [schnell]**, use:
```bash
python src/generate_images_csv.py \
  model=schnell \
  csv_path=/path/to/prompts.csv \
  loss=power \
  lambda_schedule=power \
  lambda_schedule.start=25 \
  lambda_schedule.end=1 \
  lambda_schedule.power=2 \
  output.run_name=example_run
```
For **FLUX.2 [dev]**, use:
```bash
python src/generate_csv_flux2.py \
  model=flux2dev \
  csv_path=/path/to/prompts.csv \
  loss=power \
  lambda_schedule=power \
  lambda_schedule.start=25 \
  lambda_schedule.end=1 \
  lambda_schedule.power=2 \
  output.run_name=example_run
```

## 📊 Evaluation / Metrics
This script computes quantitative evaluation metrics for generated images.  
Results are saved to `outputs/metrics/results.txt` by default and can be customized if needed.

The following metrics are computed: **CLIP-T**, **MeanArtifactFreq (%)**, **ArtifactPixelRatio (%)**, **MAE**, **MAE(A)**, **MAE(NA)**.

#### Run metric computation:

```bash
python src/generate_metrics.py \
  metrics.generated_dir=/path/to/generated/images \
  metrics.reference_dir=/path/to/reference/images \
  metrics.prompts_csv=/path/to/prompts.csv 
```
For computing **ImageReward**, please refer to the official repository: https://github.com/zai-org/ImageReward

> [!NOTE]  
> Prompt CSV files used for evaluation are provided in the `datasets/` directory.



## 🗂 Generate Custom Evaluation Dataset
Generate a dataset by searching for valid seeds and saving prompts + seeds into a CSV file.  
Prompts are provided as `.txt` files (one per line). Example files are in `prompts/`.
The script also saves generated images and corresponding artifact masks.
The `seed` parameter specifies the starting seed from which the search begins

```bash
python src/generate_dataset.py \
  model=dev \
  seed=100000 \
  dataset.prompts_file=prompts/animals.txt \
  dataset.name=my_dataset \
  output.run_name=dataset_gen
```

> [!NOTE]  
> Dataset generation is supported for **FLUX.1 [dev]**, **FLUX.1 [schnell]**, **FLUX.2 [dev]**, and **SDXL**.  
> To switch models, only the script name and the `model` value need to be changed:
> - `generate_dataset.py` → dev/schnell 
> - `generate_dataset_flux2.py` → flux2dev
> - `generate_dataset_sdxl.py` → sdxl
