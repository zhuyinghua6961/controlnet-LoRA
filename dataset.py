import os
import re
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from utils import create_heatmap_from_targets

def parse_prompt_from_text(text):
    matches = re.findall(
        r'distance\s*=\s*([+-]?\d+\.?\d*)\s*m[^,]*,\s*velocity\s*=\s*([+-]?\d+\.?\d*)\s*m/s',
        text,
        re.IGNORECASE
    )
    if not matches:
        raise ValueError("未找到 distance/velocity 对")
    return [(float(r), float(v)) for r, v in matches]

class RadarControlNetDataset(Dataset):
    def __init__(self, data_dir, tokenizer, size=512, range_max=200, vel_max=40):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.size = size
        self.range_max = range_max
        self.vel_max = vel_max

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
        with open(os.path.join(self.data_dir, base + '.txt'), 'r') as f:
            text = f.read()
        targets = parse_prompt_from_text(text)

        heatmap = create_heatmap_from_targets(
            targets,
            img_size=(self.size, self.size),
            range_max=self.range_max,
            vel_min=-self.vel_max,
            vel_max=self.vel_max
        )
        control_image = torch.from_numpy(heatmap).float().repeat(3, 1, 1)

        image = Image.open(os.path.join(self.data_dir, base + '.png')).convert("RGB")
        image = image.resize((self.size, self.size))
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

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