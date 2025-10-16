import os
import yaml
import argparse
import importlib.util


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def save_config(path, cfg):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        yaml.safe_dump(cfg, f)


def import_py(path, attr):
    spec = importlib.util.spec_from_file_location("_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return getattr(mod, attr)


def main():
    parser = argparse.ArgumentParser(description="全流程：先训 ControlNet，再 LoRA 微调 UNet")
    parser.add_argument("--controlnet_config", type=str, default="config.yaml", help="ControlNet 训练配置")
    parser.add_argument("--lora_config", type=str, default="config_lora.yaml", help="LoRA 训练配置")
    parser.add_argument("--from_scratch", action="store_true", help="ControlNet 阶段从头开始")
    args = parser.parse_args()

    # 阶段1：训练 ControlNet（调用现有脚本的 main）
    print("\n==== 阶段1：ControlNet 训练 ====")
    t1_path = os.path.join(os.path.dirname(__file__), 'train_controlnet_fixed.py')
    t1_main = import_py(t1_path, 'main')

    # 控制是否从头
    resume_flag = 'none' if args.from_scratch else None
    t1_main(args.controlnet_config, resume_flag)

    # 读取 ControlNet 配置，解析输出目录
    control_cfg = load_config(args.controlnet_config)
    control_out = control_cfg['project']['output_dir']
    controlnet_final_path = os.path.join(control_out, 'controlnet')
    if not os.path.isdir(controlnet_final_path):
        raise FileNotFoundError(f"未找到 ControlNet 最终目录: {controlnet_final_path}")
    print(f"✅ ControlNet 完成，路径: {controlnet_final_path}")

    # 阶段2：配置并训练 LoRA（调用新脚本的 main）
    print("\n==== 阶段2：UNet LoRA 微调 ====")
    lora_cfg_path = args.lora_config
    lora_cfg = load_config(lora_cfg_path)
    lora_cfg.setdefault('controlnet', {})['path'] = controlnet_final_path

    # 将更新后的 LoRA 配置保存到一个派生文件，避免覆盖原配置
    lora_cfg_out = lora_cfg_path.replace('.yaml', '_autofilled.yaml')
    save_config(lora_cfg_out, lora_cfg)
    print(f"📝 LoRA 配置已自动写入: {lora_cfg_out}")

    t2_path = os.path.join(os.path.dirname(__file__), 'train_controlnet_lora.py')
    t2_main = import_py(t2_path, 'main')
    t2_main(lora_cfg_out, None)

    print("\n🎉 全流程完成：ControlNet 训练 + UNet LoRA 微调")


if __name__ == "__main__":
    main()


