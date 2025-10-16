# 🎯 雷达 RD 图生成（Stable Diffusion + ControlNet）

本项目使用 Stable Diffusion v1.5 与 ControlNet，实现根据目标的“距离/速度”生成雷达 Range‑Doppler（RD）图。训练阶段仅学习 ControlNet 或在此基础上用 UNet LoRA 做轻量微调，从而在“位置准确”的同时提升“画质与风格”。

- 训练与使用请直接阅读：`START_LORA.md`
- 调参与问题诊断请参考：`TUNING_GUIDE.md`

---

## 🧠 原理概述

### 1) 任务与数据
- 输入：每张样本包含一张 RD 图（512×512）与对应文本，文本中含若干目标的 `distance (m)` 与 `velocity (m/s)`。
- 条件表示：将所有目标转换为二维热力图（Y=距离，X=速度），对目标位置放置高斯点并叠加，得到 1 通道热图；为适配 ControlNet，用 3 通道重复并与图像同分辨率对齐。

### 2) ControlNet 如何控制位置
- 冻结 SD1.5 的 `VAE/UNet/TextEncoder`，仅训练 `ControlNet`。
- 推理时将条件热力图输入 ControlNet，
  ControlNet 产生下/中间块残差，注入到 UNet 对应层，从而约束生成位置与结构。

### 3) 训练目标（扩散噪声预测）
- 采用 DDPM 训练范式：
  - 对图像潜空间 `latents` 加入噪声 `noise` 于随机 timestep `t`。
  - UNet 预测 `noise`（epsilon 预测）；损失为逐像素 MSE。
- 为提升“目标位置”学习效率，采用“区域加权 MSE”：
  - 依据热力图在 64×64 的潜空间对齐处生成 `target_mask`。
  - 目标区域赋更高权重、背景为 1；计算加权平均 MSE 作为优化目标。

### 4) 热力图与 Sigma
- 将物理量映射到图像坐标：
  - 距离归一化到 Y 轴（近→远）。
  - 速度归一化到 X 轴（负→正）。
- 对目标位置放置二维高斯；`sigma` 决定“点的扩散程度”。
  - 小 `sigma` → 目标更尖锐、边界清晰（定位训练更精确，但对齐要求高）。
  - 大 `sigma` → 目标更宽、更平滑（训练更稳，但易糊化）。
- 训练与推理的 `sigma` 必须保持一致。

### 5) LoRA 微调（画质增强）
- 在“已训练的 ControlNet”基础上，只对 UNet 注意力层注入 LoRA 并训练少量参数（rank 8–16）。
- ControlNet 负责“结构/位置”，LoRA 负责“风格/锐度/对比度”，二者职责互补。
- 推理时按需加载 LoRA 并设置 `lora_scale` 调整风格强度。

### 6) 推理机制
- 构造与训练一致的条件热力图；
- 使用 `StableDiffusionControlNetPipeline`：加载 SD1.5、ControlNet（可选 LoRA），设置步数/CFG/控制强度；
- 采样器（如 DDIM/Euler/DPM++）迭代还原，得到符合条件的 RD 图。

---

## 📎 文档导航
- 开始训练与推理（含一键全流程与命令）：`START_LORA.md`
- 调参与故障排查：`TUNING_GUIDE.md`

---

## 📂 主要文件
- 训练：`train_controlnet_fixed.py`（加权损失）、`train_controlnet_lora.py`（UNet LoRA）、`train_full_pipeline.py`（一键流程）
- 推理：`inference_enhanced.py`、`inference_lora.py`
- 数据与工具：`dataset.py`、`utils.py`、`generate_control_image.py`
- 配置：`config.yaml`、`config_lora.yaml`
- 指南：`START_LORA.md`、`TUNING_GUIDE.md`

---

## 📝 License
MIT
