import os
import re
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from utils import create_heatmap_from_targets

def parse_prompt_from_text(text):
    """
    解析 prompt 文本，提取目标的距离和速度信息
    支持格式：
    - target number = N, the first target: distance = 41m, velocity = -20.00m/s
    - 或旧格式：distance = 41m, velocity = -20.00m/s
    """
    # 匹配 "distance = XXm, velocity = XXm/s" 格式
    matches = re.findall(
        r'distance\s*=\s*([+-]?\d+\.?\d*)\s*m[^,]*,\s*velocity\s*=\s*([+-]?\d+\.?\d*)\s*m/s',
        text,
        re.IGNORECASE
    )
    if not matches:
        raise ValueError(f"未找到 distance/velocity 对，文本内容：{text[:100]}")
    return [(float(r), float(v)) for r, v in matches]

class RadarControlNetDataset(Dataset):
    def __init__(self, data_dir, tokenizer, size=512, range_max=200, vel_max=40, heatmap_sigma=15.0):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.size = size
        self.range_max = range_max
        self.vel_max = vel_max
        self.heatmap_sigma = heatmap_sigma
        print(f"🎯 热力图Sigma参数: {self.heatmap_sigma}")

        txt_files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
        self.basenames = []
        for f in txt_files:
            base = f[:-4]
            if os.path.exists(os.path.join(data_dir, base + '.png')):
                self.basenames.append(base)
        print(f"✅ 加载 {len(self.basenames)} 个样本")

    def __len__(self):
        return len(self.basenames)

    def __getitem__(self, idx):
        base = self.basenames[idx]
        try:
            # 读取文本文件
            txt_path = os.path.join(self.data_dir, base + '.txt')
            with open(txt_path, 'r', encoding='utf-8') as f:
                text = f.read()
            targets = parse_prompt_from_text(text)

            # 生成热力图
            heatmap = create_heatmap_from_targets(
                targets,
                img_size=(self.size, self.size),
                range_max=self.range_max,
                vel_min=-self.vel_max,
                vel_max=self.vel_max,
                sigma=self.heatmap_sigma
            )
            control_image = torch.from_numpy(heatmap).float().repeat(3, 1, 1)

            # 读取图像文件
            img_path = os.path.join(self.data_dir, base + '.png')
            image = Image.open(img_path).convert("RGB")
            image = image.resize((self.size, self.size), Image.LANCZOS)
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
            
        except FileNotFoundError as e:
            raise FileNotFoundError(f"文件不存在: {e.filename}")
        except ValueError as e:
            raise ValueError(f"文件 {base} 解析失败: {e}")
        except Exception as e:
            raise RuntimeError(f"加载样本 {base} 时出错: {e}")

        # 使用简洁统一的 prompt，与数据集描述一致
        prompt = "Radar Range-Doppler diagram with blue background and cross-shaped targets. No axes, no grid, no text."
        input_ids = self.tokenizer(
            prompt,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids[0]

        return {
            "pixel_values": image,
            "conditioning_pixel_values": control_image,
            "input_ids": input_ids
        }