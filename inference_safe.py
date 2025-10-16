"""
安全版推理脚本 - 最小改动版
只将硬编码参数改为可配置，不添加新参数
用于验证与旧模型的兼容性
"""
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
    """
    解析测试 prompt，支持多种格式：
    1. target number = N, the first target: distance = 41m, velocity = -20.00m/s
    2. target 1: distance = 143m, velocity = -20.00m/s
    3. distance = 143m, velocity = -20.00m/s (直接格式)
    """
    targets = []
    
    # 通用匹配：任何包含 distance 和 velocity 的行
    matches = re.findall(
        r'distance\s*=\s*([+-]?\d+\.?\d*)\s*m[^,]*,\s*velocity\s*=\s*([+-]?\d+\.?\d*)\s*m/s',
        prompt_text,
        re.IGNORECASE
    )
    
    for r_str, v_str in matches:
        r, v = float(r_str), float(v_str)
        targets.append((r, v))
    
    if len(targets) == 0:
        raise ValueError(f"未找到有效的 target 信息\n文本内容：{prompt_text[:200]}")
    if len(targets) > 10:
        raise ValueError(f"目标数量 {len(targets)} 超过限制（最多 10 个）")
    
    return targets

def main(args):
    config = load_config(args.config)
    range_max = config["data"]["range_max"]
    vel_max = config["data"]["vel_max"]

    print(f"📥 加载 ControlNet 模型: {args.controlnet_path}")
    controlnet = ControlNetModel.from_pretrained(args.controlnet_path, torch_dtype=torch.float16)
    
    print("📥 加载 Stable Diffusion pipeline...")
    try:
        # 尝试新版本加载方式
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            torch_dtype=torch.float16,
            safety_checker=None,
            variant="fp16",
            use_safetensors=True
        )
    except TypeError:
        # 如果失败，使用旧版本加载方式
        print("⚠️  使用兼容模式加载...")
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
    # 转换为RGB格式，保持[0,1]浮点范围，与训练时一致
    heatmap_rgb = np.stack([heatmap, heatmap, heatmap], axis=-1).astype(np.float32)
    control_image = Image.fromarray((heatmap_rgb * 255).astype(np.uint8))

    base_prompt = "Radar Range-Doppler diagram with blue background and cross-shaped targets. No axes, no grid, no text."

    print(f"🎨 推理参数:")
    print(f"  - num_inference_steps: {args.num_inference_steps}")
    print(f"  - guidance_scale: {args.guidance_scale}")
    
    # 🔴 关键：只修改为可配置参数，不添加新参数
    images = pipe(
        [base_prompt] * args.batch_size,
        image=[control_image] * args.batch_size,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale
    ).images

    base_name = os.path.splitext(os.path.basename(args.prompt_file))[0]
    for i, img in enumerate(images):
        output_path = f"{args.output_dir}/{base_name}_batch_{i+1:02d}.png"
        img.save(output_path)
        print(f"✅ 保存: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="安全版推理脚本 - 验证兼容性")
    parser.add_argument("--controlnet_path", type=str, required=True)
    parser.add_argument("--prompt_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="test_output")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--config", type=str, default="config.yaml")
    # 将硬编码改为可配置，默认值与旧版相同
    parser.add_argument("--num_inference_steps", type=int, default=20, 
                        help="推理步数（默认20，与旧版相同）")
    parser.add_argument("--guidance_scale", type=float, default=7.0, 
                        help="引导强度（默认7.0，与旧版相同）")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    main(args)

