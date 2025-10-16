# 🔧 调参与故障排查指南（ControlNet + LoRA）

面向“雷达 RD 图生成”任务，提供一套按优先级排序的调参要点与常见问题排查表。

---

## 1. 先看哪些指标？
- 训练（TensorBoard）：
  - `Loss/target_loss`（目标区 MSE）应稳步下降
  - `Loss/weighted_loss`（优化目标）随训练下降
  - `Loss/weight_mean` ≈ 1 + (w-1)*target_ratio（检查权重是否合理）
- 推理：目标位置误差（像素）、十字锐度、背景纯净度（无纹理/偏色）

---

## 2. 画质差但位置准
- 降低热图扩散：`heatmap_sigma` 从 15 → 8（训练与推理一致）
- 推理参数：steps 30–50、CFG 3–5、controlnet_conditioning_scale 1.0–1.3
- 训练权重：`loss_weight_factor` 20–50（过高会糊）
- 加 LoRA：仅训 UNet LoRA，推理 `lora_scale` 0.6–0.8

### 2.1 保持权重100 + 用 LoRA 提升画质（可行方案）
- 适用：你已验证权重100能保证“位置极准”，担心降权后定位变差。
- 做法：
  1) 保持 ControlNet 训练/或既有模型不变（`loss_weight_factor=100`）。
  2) 仅训练 UNet LoRA（rank 8–16，lr 5e-5，epoch 3–5）。
  3) 推理加载 LoRA，`lora_scale` 0.6–0.8；`controlnet_conditioning_scale` 1.1–1.3；CFG 3–5；steps 30–50。
- 监控：
  - 定位是否保持（抽样计算像素误差）；
  - 锐度/对比度是否提升（可用 Sobel/梯度能量或人工评估）。
- 风险与化解：
  - 风险：LoRA 过强可能引入纹理/色偏，极端时轻微影响结构。
  - 化解：先降 `lora_scale`（如 0.5–0.6）；若仍不稳，再小幅提高 `controlnet_conditioning_scale`（如 1.3）。

---

## 3. 位置不准
- 提高权重（在可控范围内）：`loss_weight_factor` 50–150
- 增大 `controlnet_conditioning_scale` 1.2–1.5
- 保证热图与图像严格对齐（尺寸/坐标映射/归一化）
- 延长 warmup、降低学习率峰值，增强稳定性

---

## 4. Loss 抖动或长期不降
- 学习率过大：降一档（如 1e-5 → 5e-6），或加长 `lr_warmup_steps`
- 权重过强：`loss_weight_factor` 下调至 20–50
- 精度与稳定：fp16→bf16（若硬件支持）、或临时全精度验证
- 数据问题：抽样可视化热图与 RD 图，核对目标是否匹配

---

## 5. LoRA 导致结构偏移
- 减小 `lora_scale`（如 0.7 → 0.5）
- 训练 LoRA 的 epoch 减少至 3–5；学习率 5e-5
- 提高 `controlnet_conditioning_scale` 以加强结构约束

---

## 6. 推荐的组合
- 位置优先：`sigma=8`，`loss_weight_factor=80`，`controlnet_scale=1.3`，无 LoRA
- 画质优先：`sigma=8`，`loss_weight_factor=30`，`controlnet_scale=1.2` + LoRA(`lora_scale=0.7`)
- 平衡方案：`sigma=8`，`loss_weight_factor=50`，`controlnet_scale=1.2` + LoRA(`0.6`)
- 定位极准优先（你提的方案）：`sigma=8`，`loss_weight_factor=100`（训练/既有），推理时加 LoRA(`0.6–0.8`)；必要时 `controlnet_scale=1.3`

---

## 7. 常见 Q&A
- Q: target_ratio 很小（<0.5%）怎么办？
  - A: 调低阈值或对热图做轻微膨胀，减小稀疏度；或降低权重。
- Q: 背景偏紫/有纹理？
  - A: 降 CFG；降低 `lora_scale`；增强 prompt 中的“solid dark blue, no texture”。
- Q: 训练很慢？
  - A: 增大 `dataloader_num_workers`；开启 xformers；用更大的梯度累积。

---

## 8. 最小排查清单
- 训练/推理 `sigma` 是否一致？
- ControlNet 条件与图像是否一一匹配？
- 学习率与 warmup 是否过激？
- 权重是否过高导致糊化？
- LoRA 是否过强（调低 `lora_scale`）？

祝你训练顺利！
