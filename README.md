# 💎 DIAMOND: Directed Inference for Artifact Mitigation in Flow Matching Models

<div align="center">

🌐 **[Project Page](https://gmum.github.io/DIAMOND/)** &nbsp;&nbsp;|&nbsp;&nbsp; 📄 **[arXiv](https://arxiv.org/abs/2602.00883)**

<br>

Alicja Polowczyk*, Agnieszka Polowczyk*, Piotr Borycki, Joanna Waczyńska,  
Jacek Tabor, Przemysław Spurek  
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

