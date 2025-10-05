import argparse
import yaml
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DDIMScheduler
from PIL import Image
from utils import create_heatmap_from_targets
import numpy as np
import re
import os

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def parse_test_prompt(prompt_text):
    targets = []
    pattern = r'target\s+(\d+)\s*:\s*distance\s*=\s*([+-]?\d+\.?\d*)\s*m\s*,\s*velocity\s*=\s*([+-]?\d+\.?\d*)\s*m/s'
    matches = re.findall(pattern, prompt_text, re.IGNORECASE)
    matches.sort(key=lambda x: int(x[0]))
    for _, r_str, v_str in matches:
        r, v = float(r_str), float(v_str)
        targets.append((r, v))
    if len(targets) == 0:
        raise ValueError("未找到有效的 target 行")
    if len(targets) > 10:
        raise ValueError(f"目标数量 {len(targets)} 超过限制（最多 10 个）")
    return targets

def main(args):
    config = load_config(args.config)
    range_max = config["data"]["range_max"]
    vel_max = config["data"]["vel_max"]

    controlnet = ControlNetModel.from_pretrained(args.controlnet_path, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")

    with open(args.prompt_file, 'r') as f:
        prompt_text = f.read()
    
    targets = parse_test_prompt(prompt_text)
    print(f"🎯 解析到 {len(targets)} 个目标: {targets}")

    heatmap = create_heatmap_from_targets(
        targets,
        img_size=(512, 512),
        range_max=range_max,
        vel_min=-vel_max,
        vel_max=vel_max
    )
    control_image = Image.fromarray((heatmap * 255).astype(np.uint8)).convert("RGB")

    base_prompt = "Radar Range-Doppler diagram with blue background and cross-shaped targets. No axes, no grid, no text."

    images = pipe(
        [base_prompt] * args.batch_size,
        image=[control_image] * args.batch_size,
        num_inference_steps=20,
        guidance_scale=7.0
    ).images

    base_name = os.path.splitext(os.path.basename(args.prompt_file))[0]
    for i, img in enumerate(images):
        output_path = f"{args.output_dir}/{base_name}_batch_{i+1:02d}.png"
        img.save(output_path)
        print(f"✅ 保存: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--controlnet_path", type=str, required=True)
    parser.add_argument("--prompt_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="test_output")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    main(args)