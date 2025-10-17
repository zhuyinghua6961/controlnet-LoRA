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

### 5.5 两阶段 + 轻联合训练（原理）
- 动机：直接“强联合”易相互干扰、定位收敛慢。将“位置学习（ControlNet）”与“画质微调（UNet LoRA）”解耦，可稳定、快速地获得“位置稳 + 画质好”。
- 职责分离：
  - 阶段1：仅训练 ControlNet（UNet 主干/LoRA/VAEs/文本编码器冻结），用加权 MSE 强化目标区域，先把“位置学稳”。
  - 阶段2：轻联合——继续训练 ControlNet，同时只训练 UNet 的 LoRA 层（主干仍冻结），以较小代价提升锐度、对比度与颜色稳定。
- 参数分组与学习率：
  - ControlNet 使用小 LR（如 1e-6），维持结构；LoRA 使用较大 LR（如 5e-5）以快速注入风格；分别作为两个参数组优化。
  - 其余模块（VAE/TextEncoder/UNet 主干）保持冻结，避免大范围参数漂移。
- 训练策略：
  - 可在阶段2启用“渐进 LoRA”：前若干 epoch 令 `lora_lr=0` 或很小，随后线性升至目标 LR；或在推理时控制 `lora_scale`。
  - 建议阶段2适度降低 `loss_weight_factor`（如从 100 降到 60），在不损伤定位的前提下提升锐度。
  - 启用早停（监控 `weighted_loss_epoch`）与梯度累计（扩大有效 batch）以提升稳定性与收敛效率。
- 监控与权衡：
  - 关注 `target_loss/base_loss/bg_loss` 是否继续下降或持平；
  - 可视化目标位置误差（px）、十字锐度/对比度、背景纯净度（是否引入纹理/偏色）。
  - 若 LoRA 影响结构：降低 `lora_scale` 或增大 `controlnet_conditioning_scale`（推理时）。

### 优化的加权 Loss 函数（训练中使用）
- 目标：在稀疏目标场景下强化“目标区域”的学习，提升位置精度与收敛稳定性。
- 基础损失：逐像素 MSE
  \[ L_{base} = \|\hat{\epsilon} - \epsilon\|_2^2 \]
- 权重图构造（由条件热图得到）：
  - 灰度化：`heatmap_gray = mean(cond, dim=channel)`（B×1×512×512）
  - 下采样：使用 Max Pooling（kernel/stride=8）至 64×64（保持峰值，不被平均稀释）
  - 二值阈值：`target_mask = (weight_map > 0.1)`，目标/背景清晰划分
  - 赋权：目标区 = `weight_factor`（如 30–100），背景 = 1
  - 可选 Focal：对高误差区域追加动态权重，关注难样本
- 加权平均：
  \[ L = \frac{\sum (L_{base} \cdot W)}{\sum W} \]
- 记录指标（TensorBoard）：
  - `weighted_loss`、`base_loss`、`target_loss`、`bg_loss`
  - `weight_mean`（有效平均权重）、`target_ratio`（目标像素占比）
- 关键收益：
  - Max Pooling 保峰值，避免 bilinear 下采样的峰值稀释
  - 二值权重直观可控，目标/背景分离清晰
  - 加权平均使数值与批次稳定，便于对比与调参
  - 显著提升位置精度，对 RD 十字目标尤为有效

#### 对比：基础 MSE vs 朴素加权 vs 优化加权（当前）

| 项目 | 定义/实现 | 下采样/掩码 | 归一化 | 优缺点与适用 |
|---|---|---|---|---|
| 基础 MSE | \(mean(L_{base})\) | 无 | 无 | 简单稳定；目标稀疏时易被背景主导，位置精度差 |
| 朴素加权 | \(mean(L_{base} \cdot W)\) | 常用 bilinear 下采样热图直接当权重 | 一般为对像素数均值（非按权重和归一） | 简单易用；但峰值被稀释、数值随目标占比波动，易不稳或过拟合背景 |
| 优化加权（当前） | \(\frac{\sum(L_{base} \cdot W)}{\sum W}\) | Max Pool 至 64×64 + 二值掩码；可选 Focal | 按权重和归一（加权平均） | 保峰值、强调目标且数值稳定；位置精度高、收敛更稳；实现略复杂 |

> 注：若使用 朴素加权，建议至少改为“按权重和做归一”（即本表第三列的做法），并避免 bilinear 稀释峰值。

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
