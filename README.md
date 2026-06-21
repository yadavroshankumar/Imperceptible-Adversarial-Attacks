# From Pixels to Policy: LLM-Controlled DeepFool Attacks Balancing Stealth and Efficiency via Grad-CAM Spatial Constraints

This repository contains the full experiment code, attack pipelines, and results for a comparative study of four adversarial attack strategies — built on the **DeepFool** algorithm — evaluated against a ResNet-based cat/dog image classifier. The work introduces a novel **LLM-as-agent framework** that dynamically tunes attack hyperparameters round-by-round based on real-time Grad-CAM and SSIM feedback.

> Roshan Kumar Yadav, Bharath K. Samanthula — School of Computing, Montclair State University

---

## Overview

Deep neural networks achieve human-level accuracy on benchmark vision tasks yet remain fragile to imperceptible adversarial perturbations. This project systematically compares **four progressively intelligent attack strategies**, all grounded in DeepFool, to study how *spatial awareness* and *adaptive hyperparameter control* affect the trade-off between attack success and perceptual stealth.

All attacks are evaluated under a strict **SSIM ≥ 0.90** constraint, ensuring that any reported "success" reflects a genuinely imperceptible adversarial example — not just a misclassification.

### Experimental Variants

| Variant | Strategy | Description |
|---|---|---|
| **E1** | Full-Image Attack | Baseline DeepFool applied uniformly across every pixel, no spatial restriction. |
| **E2** | Random Spatial Mask | Perturbation confined to a randomly selected region covering 25% of the image. |
| **E3** | Grad-CAM Guided Attack | Perturbation restricted to regions the model's own Grad-CAM heatmap identifies as decision-critical. |
| **E4** | LLM-Agent Adaptive Attack | A Large Language Model dynamically selects epsilon, overshoot, Grad-CAM threshold, and iteration budget *per image, per round*, with full natural-language reasoning logged for every decision. |

### Key Findings

- **Semantic spatial awareness is the single most important factor** for producing imperceptible yet effective adversarial examples. Grad-CAM-guided (E3) and LLM-agent-guided (E4) attacks consistently outperform spatially blind approaches (E1, E2) once perceptibility is enforced.
- **E3 achieves the highest SSIM-constrained success rate (49%)** among the three fixed-parameter strategies — proof that concentrating perturbation in the model's true attention region is more efficient than perturbing everything or perturbing randomly.
- **E4's LLM agent provides transparent, auditable reasoning** for every hyperparameter decision, demonstrating that language models can act as interpretable controllers for numerical optimization loops — not merely as text generators.
- A **98%-accurate classifier** — representative of production-grade systems — was shown to be thoroughly vulnerable to all four attack strategies, challenging the common assumption that high test accuracy implies adversarial robustness.

---

## Results Summary

Quantitative comparison across 100 held-out test images, under the SSIM ≥ 0.90 perceptibility constraint:

| Strategy | Mean Iterations | ASR (SSIM ≥ 0.90) | Mean SSIM |
|---|---|---|---|
| E1: Full-Image | 23 | 19% | ~0.9514 |
| E2: Random Mask | 240 | 44% | ~0.9502 |
| **E3: Grad-CAM** | 177 | **49%** | ~0.9566 |
| E4: LLM-Agent | 218 | 43% | ~0.9404 |

Model baseline: **ResNet (ImageNet pretrained)**, transfer-learned on a balanced 25,000-image cat/dog dataset → **100% train accuracy / 98% test accuracy**.

---

## Repository Structure

```
.
├── data/                          # Raw datasets (gitignored — see "Large Files" below)
├── results/                       # Experiment outputs, logs, generated images (gitignored)
├── experiments/
│   ├── e1_full_image/             # Baseline full-image DeepFool attack
│   ├── e2_random_mask/            # Random 25% region masking
│   ├── e3_gradcam/                # Grad-CAM guided critical-region attack
│   └── e4_llm_agent/              # LLM-agent adaptive attack pipeline
├── Adversarial_Dog_AIAgent/       # Successful E4 adversarial outputs
├── agent_decisions.jsonl          # Full LLM reasoning + parameter trace (E4)
├── agent_attack_log.csv           # Per-image E4 attack statistics
├── requirements.txt
├── .env.example
└── README.md
```

---

## Methodology

All four strategies share the same DeepFool core loop. Given classifier $f$ and original image $x_0$, at each iteration $i$ the gradient $\nabla f(x_i)$ is computed and the minimal perturbation step is derived as:

```
r_i = -(f(x_i) - 0.5) / ||∇f(x_i)||² · ∇f(x_i)
```

The cumulative perturbation is clipped to $[-\varepsilon, +\varepsilon]$ per channel, and the image is updated as $x_{i+1} = \text{clip}(x_0 + \eta \cdot r_i, 0, 1)$.

- **E1** applies $r_i$ unconstrained across all pixels ($\varepsilon = 0.0314$, $\eta = 1.02$, max 1,000 iterations).
- **E2** restricts $r_i$ to a random binary mask covering 25% of the image.
- **E3** restricts $r_i$ to a Grad-CAM heatmap thresholded at 0.40, concentrating perturbation only where the model's decision is actually grounded.
- **E4** runs up to 6 sequential rounds per image. Before each round, an LLM agent receives the current confidence, distance from the decision boundary, and a full log of prior round outcomes (success/failure, SSIM, iterations consumed). The agent then outputs a JSON object selecting $\varepsilon \in [0.01, 0.30]$, $\eta \in [1.00, 1.20]$, Grad-CAM threshold $\in [0.10, 0.60]$, and max iterations $\in [50, 500]$ — along with a natural-language justification for the choice, logged to `agent_decisions.jsonl`.

Every successful adversarial example must satisfy **both** misclassification **and** SSIM ≥ 0.90 — failing either condition discards the result and triggers parameter adjustment (E4) or termination (E1–E3).

---

## Quick Setup

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\activate      # Windows PowerShell
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API keys

Copy `.env.example` to `.env` and add your `OPENAI_API_KEY` (required for the E4 LLM-agent experiments).

> ⚠️ **Never commit `.env`** — it is already excluded via `.gitignore`.

### 4. Run an experiment

```bash
# Baseline full-image attack
python experiments/e1_full_image/run.py

# Grad-CAM guided attack
python experiments/e3_gradcam/run.py

# LLM-agent adaptive attack
python experiments/e4_llm_agent/run.py
```

Each script reads test images from `data/`, runs the corresponding attack pipeline, and writes adversarial outputs + per-image CSV logs to `results/`.

---

## Dataset

Experiments use the **Dog and Cat Classification Dataset** (25,000 images, balanced 12,500/12,500 split), publicly available on Kaggle:

🔗 [kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset](https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset)

Download the dataset separately and place it under `data/` — it is **not** included in this repository (see large file guidance below).

---

## Large Files & Git LFS

Do **not** commit large model or dataset files (e.g. `data/cat_dog_classifier_resnet150.h5`) directly to the repository. If you need to version-control them, use Git LFS:

```bash
git lfs install
git lfs track "data/*.h5"
git add .gitattributes
```

By default, `.gitignore` excludes `/data/` and `/results/`. Keep raw datasets and generated outputs in those folders so they stay out of version control.

If you plan to share trained models, prefer releasing them via cloud storage (Google Drive, S3) and link download instructions here rather than committing binaries.

---

## Pushing to a New Repository

```bash
git init
git add .
git commit -m "Initial commit - cleaned and added repo metadata"
git remote add origin <your-repo-url>
git branch -M main
git push -u origin main
```

> Before pushing, review `experiments/` and `results/` and remove or relocate any large result artifacts.

---

## Environment Notes

This code is built around **TensorFlow** (CPU or GPU builds). If you have a CUDA-compatible GPU, install the TensorFlow package matching your CUDA/cuDNN version for significantly faster Grad-CAM and DeepFool gradient computation.

The E4 LLM-agent pipeline requires network access to the OpenAI API and introduces additional latency per attack round — expect E4 experiments to take noticeably longer than E1–E3.

---

## Applications

Beyond serving as attack benchmarks, the adversarial examples generated by this pipeline support three practical use cases:

1. **Security evaluation** — exposing concrete vulnerabilities in deployed vision systems (autonomous vehicles, facial recognition, medical imaging) that rely on architectures similar to this ResNet classifier.
2. **Adversarial training** — incorporating SSIM-constrained adversarial examples into training data to improve model robustness without introducing visually obvious corrupted inputs.
3. **Data augmentation** — in low-data regimes, E3/E4-generated examples offer a semantically coherent augmentation strategy beyond standard cropping or color jitter.

---

## Limitations

- Experiments are limited to a single binary classification task (cat vs. dog); generalization to multi-class or non-vision modalities is unvalidated.
- The E4 LLM agent depends on an external API, introducing latency unsuitable for time-critical deployment.
- The fixed SSIM ≥ 0.90 threshold is a simplification — human perceptual sensitivity varies by image content.
- The dataset, while publicly and ethically sourced, means findings should be interpreted alongside the broader ethical considerations of adversarial tooling in real-world deployment.

---

## Future Work

- Extend to multi-class classifiers and additional model architectures (transferability analysis).
- Explore physically realizable adversarial patches.
- Incorporate E3/E4 adversarial examples directly into adversarial training protocols.


---

## References

Key prior work this project builds on:

- Moosavi-Dezfooli et al., *DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks*, CVPR 2016.
- Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization*, ICCV 2017.
- He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016.
- Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models*, ICLR 2023.

Full reference list available in the accompanying paper (`paper/main.tex`).

---

## License

This project is intended for academic and research purposes. Dataset usage follows the [Kaggle dataset license](https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset). Please review applicable terms before commercial use.#   I m p e r c e p t i b l e - A d v e r s a r i a l - A t t a c k s  
 