# 雷达 RD 图生成 - ControlNet 方案

## 依赖安装

```bash
pip install -r requirements.txt
```

## 训练

```bash
python train.py --config config.yaml
```

## 从checkpoint恢复训练

```bash
python train.py --config config.yaml --resume_from_checkpoint output/radar_controlnet/checkpoint-1000
```

## 测试

```bash
python inference.py \
  --controlnet_path output/radar_controlnet/controlnet \
  --prompt_file test_prompt.txt \
  --output_dir results \
  --batch_size 4
```

## 测试用prompt格式

target 1: distance = 143m, velocity = -20.00m/s
target 2: distance = 80m, velocity = 5.00m/s