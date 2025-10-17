# 🧩 LoRA 全流程指南（ControlNet → UNet LoRA → 推理）

本指南在“已完成 ControlNet 训练并具备较高位置精度”的前提下，通过给 UNet 注入 LoRA 来提升画质（锐度、对比度、颜色稳定）。

---

## 0. 环境与依赖

```bash
pip install -r requirements.txt
```

要求：Python 3.8+、CUDA 11.7+、显存≥12GB（推荐24GB）。

---

## 1. 阶段一：训练 ControlNet（若已训练可跳过）

使用你现有的配置：
```bash
python train_controlnet_fixed.py --config config.yaml --from_scratch
# 训练输出最终在: output/controlnet/controlnet/
```

训练过程中会在 `output/controlnet/` 产生 checkpoint 与日志。

---

## 2. 阶段二：训练 UNet LoRA（仅训练 LoRA，冻结 ControlNet/UNet主干/VAE/TextEncoder）

示例配置：`config_lora.yaml`
```yaml
project:
  name: "radar_controlnet_lora"
  output_dir: "output/controlnet_lora"

data:
  data_dir: "dataset"
  resolution: 512
  range_max: 200
  vel_max: 40

training:
  train_batch_size: 2
  gradient_accumulation_steps: 4
  num_train_epochs: 5
  learning_rate: 5.0e-5
  lr_scheduler: "cosine"
  lr_warmup_steps: 500
  max_grad_norm: 1.0
  save_steps: 2000
  mixed_precision: "fp16"
  dataloader_num_workers: 4

  # LoRA & loss
  lora_rank: 16
  heatmap_sigma: 8.0
  loss_weight_factor: 30.0
  use_focal_loss: false

controlnet:
  path: "output/controlnet/controlnet"  # 已训练好的 ControlNet
```

启动训练：
```bash
python train_controlnet_lora.py --config config_lora.yaml
# LoRA 最终权重在: output/controlnet_lora/lora/
```

建议超参：
- lora_rank: 8–16；learning_rate: 5e-5 ~ 1e-4；warmup: 500 steps；epochs: 3–10
- heatmap_sigma 建议与推理一致（如 8.0）
- loss_weight_factor 取 20–40 更温和，减少糊化

---

## 2.5 两阶段 + 轻联合训练（推荐进阶）

当你希望在定位已稳定的前提下进一步增强画质，可使用“两阶段+轻联合”脚本：

```bash
python train_joint_lora_controlnet.py --config config_joint.yaml
# 最终保存：output/joint_lora_cn/final/controlnet/ 与 output/joint_lora_cn/final/lora/
```

配置示例：`config_joint.yaml`（包含 stage1/2）
- 阶段1：仅训 ControlNet（定位优先），如 `loss_weight_factor: 100`, `heatmap_sigma: 8.0`
- 阶段2：ControlNet + UNet LoRA 联合（小 lr 微调画质），如 `lora_learning_rate: 5e-5`, `loss_weight_factor: 60`

建议：
- 数据量 3000–4000 张时，阶段1 40–80 epoch（或按你经验 80–100 更稳，配合早停）；阶段2 3–10 epoch（可按需追加）

---

## 3. 推理（支持是否加载 LoRA）

脚本：`inference_lora.py`

仅用 ControlNet：
```bash
python inference_lora.py \
  --controlnet_path output/controlnet/controlnet \
  --prompt_file test_prompt_example.txt \
  --output_dir results_cn \
  --config config.yaml \
  --lora_dir "" \
  --steps 40 --cfg 4.0 --controlnet_scale 1.2
```

ControlNet + LoRA（推荐）：
```bash
python inference_lora.py \
  --controlnet_path output/controlnet/controlnet \
  --prompt_file test_prompt_example.txt \
  --output_dir results_cn_lora \
  --config config.yaml \
  --lora_dir output/controlnet_lora/lora \
  --lora_scale 0.7 \
  --steps 40 --cfg 4.0 --controlnet_scale 1.2
```

提示：`lora_scale` 过大会影响结构细节，建议 0.6–0.8 区间微调。

---

## 4. 一键全流程（先 ControlNet → 再 LoRA）

脚本：`train_full_pipeline.py`
```bash
python train_full_pipeline.py \
  --controlnet_config config.yaml \
  --lora_config config_lora.yaml

# 强制 ControlNet 从头
python train_full_pipeline.py --controlnet_config config.yaml --lora_config config_lora.yaml --from_scratch
```

该脚本会：
1) 运行 ControlNet 训练，产出 `output/controlnet/controlnet/`
2) 自动把路径写入 LoRA 配置副本 `config_lora_autofilled.yaml`
3) 继续运行 LoRA 训练，产出 `output/controlnet_lora/lora/`

---

## 5. 早停与梯度累计（强烈推荐）

- 梯度累计
  - 通过配置 `training.gradient_accumulation_steps` 设置（LoRA 与联合脚本均已内置，使用 accelerate 管理）；有效 batch = `train_batch_size × gradient_accumulation_steps`
- 早停（防止过训、节省时间）
  - 在配置中开启：
    ```yaml
    early_stopping:
      enabled: true
      patience: 5        # 连续若干 epoch 无明显改进则停止
      min_delta: 0.0     # 认为“有改进”的最小幅度
    ```
  - 监控指标为 `weighted_loss_epoch`（训练集均值）。如需更严格，可替换为验证集指标（可扩展）。

---

## 6. 小贴士（画质与精度兼顾）
- 位置已稳时，LoRA 主要提升画质；若定位轻微受影响，降低 `lora_scale`（如 0.5–0.6）。
- 训练/推理的 `sigma` 要一致（建议 8.0）。
- 推理建议：steps 30–50、CFG 3–5、controlnet_conditioning_scale 1.0–1.3。

---

## 7. 产物与日志位置
- ControlNet 最终模型：`output/controlnet/controlnet/`
- LoRA 最终权重：`output/controlnet_lora/lora/`
- 日志：各自目录下 `logs/`

祝训练顺利，画质与精度双赢！
