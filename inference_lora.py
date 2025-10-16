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
    matches = re.findall(
        r'distance\s*=\s*([+-]?\d+\.?\d*)\s*m[^,]*,\s*velocity\s*=\s*([+-]?\d+\.?\d*)\s*m/s',
        prompt_text,
        re.IGNORECASE
    )
    targets = [(float(r), float(v)) for r, v in matches]
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
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None,
        variant="fp16",
        use_safetensors=True
    )

    # 调度器与设备
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")

    # 加载 UNet LoRA（可选）
    if args.lora_dir and os.path.isdir(args.lora_dir):
        print(f"🧩 加载 UNet LoRA: {args.lora_dir}")
        # 优先使用 pipeline 的加载与缩放接口（diffusers>=0.20）
        try:
            pipe.load_lora_weights(args.lora_dir)
            if hasattr(pipe, 'set_adapters'):
                pipe.set_adapters(["default"], [args.lora_scale])
                print(f"   - 使用 set_adapters 设置 lora_scale={args.lora_scale}")
            else:
                print("   - 当前 diffusers 不支持 set_adapters，尝试退回到直接注入处理器")
        except Exception:
            # 兼容路径：直接把注意力处理器注入到 UNet（不一定支持 scale）
            pipe.unet.load_attn_procs(args.lora_dir)
            print("   - 已通过 unet.load_attn_procs 加载（可能不支持 lora_scale）")

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
    heatmap_rgb = np.stack([heatmap, heatmap, heatmap], axis=-1).astype(np.float32)
    control_image = Image.fromarray((heatmap_rgb * 255).astype(np.uint8))

    base_prompt = "Radar Range-Doppler diagram with solid dark blue background, sharp thin cross targets, high-contrast, no blur, no grid, no axes, no text."

    images = pipe(
        [base_prompt] * args.batch_size,
        image=[control_image] * args.batch_size,
        num_inference_steps=args.steps,
        guidance_scale=args.cfg,
        controlnet_conditioning_scale=args.controlnet_scale
    ).images

    base_name = os.path.splitext(os.path.basename(args.prompt_file))[0]
    os.makedirs(args.output_dir, exist_ok=True)
    for i, img in enumerate(images):
        output_path = f"{args.output_dir}/{base_name}_lora_{i+1:02d}.png"
        img.save(output_path)
        print(f"✅ 保存: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--controlnet_path", type=str, required=True)
    parser.add_argument("--prompt_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results_lora")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--lora_dir", type=str, default="output/controlnet_lora/lora")
    parser.add_argument("--lora_scale", type=float, default=0.7)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--cfg", type=float, default=4.0)
    parser.add_argument("--controlnet_scale", type=float, default=1.2)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    main(args)


