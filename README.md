# PixelVLA: Advancing Pixel-level Understanding in Vision-Language-Action Model

[![arXiv](https://img.shields.io/badge/arXiv-2511.01571-b31b1b.svg)](https://arxiv.org/abs/2511.01571)
[![Project Page](https://img.shields.io/badge/Project-Page-blue.svg)](https://wenqiliang.github.io/PixelVLA/)
[![GitHub](https://img.shields.io/badge/GitHub-PixelVLA-black.svg)](https://github.com/WenqiLiang/PixelVLA)

Official repository for **PixelVLA**, a vision-language-action model that advances **pixel-level understanding** and **multimodal prompting** for robotic manipulation.

<p align="center">
  <img src="assets/teaser.png" alt="PixelVLA teaser" width="92%">
</p>

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Fine-Tuning with OpenVLA via LoRA](#fine-tuning-with-openvla-via-lora)
- [Method](#method)
  - [1. Visual Prompt-aware Encoder](#1-visual-prompt-aware-encoder)
  - [2. Multiscale Pixel-aware Encoder](#2-multiscale-pixel-aware-encoder)
  - [3. Continuous Action Decoder](#3-continuous-action-decoder)
- [Pixel-160K Dataset](#pixel-160k-dataset)
- [Main Results](#main-results)
- [TODO](#todo)
- [Acknowledgement](#acknowledgement)
- [Citation](#citation)

## Overview

Vision-Language-Action models (VLAs) have recently shown strong generalization and instruction-following ability for robotic control. However, most existing VLAs still operate mainly at the **image level** and rely heavily on **text-only prompts**, which limits fine-grained spatial reasoning and flexible human-robot interaction.

**PixelVLA** addresses these limitations by introducing:

- a **Visual Prompt-aware Encoder** for points, lines, regions, and masks
- a **Multiscale Pixel-aware Encoder** for injecting pixel-level spatial information
- a **Continuous Action Decoder** for precise 7D robot action prediction

In addition, we propose a two-stage automated annotation pipeline to build **Pixel-160K**, a large-scale visuomotor instruction-tuning dataset with pixel-level annotations and visual prompts.

## Installation

> PixelVLA training code is not released yet. The setup below provides a simple **OpenVLA-based reference environment** for baseline experiments.

Recommended environment:

- Python 3.10
- PyTorch 2.2.x
- `transformers 4.40.1`
- `tokenizers 0.19.1`
- `timm 0.9.10`
- `flash-attn 2.5.5`

```bash
# 1) Create environment
conda create -n pixelvla python=3.10 -y
conda activate pixelvla

# 2) Install PyTorch (example only; choose the right CUDA version for your machine)
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y

# 3) Clone and install OpenVLA
git clone https://github.com/openvla/openvla.git
cd openvla
pip install -e .

# 4) Install Flash-Attention 2
pip install packaging ninja
ninja --version; echo $?
pip install "flash-attn==2.5.5" --no-build-isolation
```

Notes:

- If you only need inference or lightweight experiments, a minimal dependency setup is usually enough.
- Later versions of `transformers`, `timm`, or `tokenizers` may introduce compatibility issues, so version pinning is recommended.

## Fine-Tuning with OpenVLA via LoRA

> Until official PixelVLA training code is released, you can use **OpenVLA's LoRA pipeline** as a practical baseline.

Main script:

```bash
vla-scripts/finetune.py
```

### 1. Prepare data

Example: BridgeData V2

```bash
cd <PATH TO DATA ROOT>

wget -r -nH --cut-dirs=4 --reject="index.html*" \
  https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/

mv bridge_dataset bridge_orig
```

### 2. Launch LoRA fine-tuning

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir <PATH TO DATA ROOT> \
  --dataset_name bridge_orig \
  --run_root_dir <PATH TO OUTPUT DIR> \
  --adapter_tmp_dir <PATH TO TEMP DIR> \
  --lora_rank 32 \
  --batch_size 16 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug <True or False> \
  --save_steps <SAVE STEPS> \
  --wandb_project <PROJECT> \
  --wandb_entity <ENTITY>
```

### 3. Quick notes

- `batch_size=16` and `grad_accumulation_steps=1` need roughly **72 GB** GPU memory.
- On smaller GPUs, reduce `batch_size` and increase `grad_accumulation_steps`.
- For multi-GPU training, set `--nproc-per-node` to the number of GPUs.
- On BridgeData V2, using `--image_aug False` may lead to very high training `action_accuracy`, since OpenVLA was already pretrained on overlapping data.

### 4. Using your own dataset

Two common options:

- **Recommended:** convert the dataset to **RLDS** format and register the dataset config / transform in the codebase.
- **Alternative:** implement your own custom PyTorch `Dataset` in the fine-tuning pipeline.

## Method

PixelVLA extends a VLA backbone with three key components:

### 1. Visual Prompt-aware Encoder

Encodes user-provided prompts such as:

- points
- lines
- bounding regions
- masks

This allows the model to preserve image-space spatial cues and respond to richer instructions than text alone.

### 2. Multiscale Pixel-aware Encoder

Extracts multi-level visual features and injects pixel-aware information into the token stream, enabling fine-grained grounding for robotic manipulation.

### 3. Continuous Action Decoder

Instead of relying only on discretized action tokens, PixelVLA predicts continuous 7D robot actions from LLM hidden states for more precise control.

<p align="center">
  <img src="assets/architecture.png" alt="PixelVLA architecture" width="92%">
</p>

## Pixel-160K Dataset

To support pixel-level visuomotor tuning, we build **Pixel-160K**, a new dataset containing:

- **160K** robot manipulation episodes
- **6.5M** image-text-action triplets
- pixel-level masks
- multimodal visual prompts

The dataset is generated by a two-stage automated annotation pipeline:

1. **Gripper-aware region proposal**
2. **Multimodal object segmentation**

<p align="center">
  <img src="assets/dataset.png" alt="Pixel-160K overview" width="92%">
</p>

## Main Results

<h3 style="margin:28px 0 14px; font-size:1.2rem;">Selected Tables from the Paper</h3>

<div class="appendix-grid">
  <div class="figure-card">
    <img src="assets/table_google_robot.png" alt="Table 1 Google Robot results" />
  </div>
  <div class="figure-card">
    <img src="assets/table_widowx.png" alt="Table 2 WidowX results" />
  </div>
  <div class="figure-card">
    <img src="assets/table_libero.png" alt="Table 3 LIBERO benchmark results" />
  </div>
</div>

## TODO

- [ ] Release training code
- [ ] Release evaluation code
- [ ] Release checkpoints
- [ ] Release data processing scripts
- [ ] Provide instructions for reproducing benchmark results

## Acknowledgement

PixelVLA builds on and is inspired by prior open-source VLA efforts, especially:

- **OpenVLA**: https://github.com/openvla/openvla
- **π0 / OpenPI**: https://github.com/Physical-Intelligence/openpi

We thank the authors and open-source contributors of these projects for advancing research on vision-language-action models.

## Citation

If you find this project useful, please cite:

```bibtex
@inproceedings{liang2026pixelvla,
  title={PixelVLA: Advancing Pixel-level Understanding in Vision-Language-Action Model},
  author={Liang, Wenqi and Sun, Gan and He, Yao and Dong, Jiahua and Dai, Suyan and Laptev, Ivan and Khan, Salman and Cong, Yang},
  booktitle={International Conference on Learning Representations},
  year={2026}
}
```
