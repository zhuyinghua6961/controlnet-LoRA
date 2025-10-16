import os
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator
from diffusers import ControlNetModel, AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.optimization import get_scheduler
try:
    from diffusers.models.attention_processor import LoRAAttnProcessor
except Exception:
    # 兼容部分版本：LoRAAttnProcessor2_0
    from diffusers.models.attention_processor import LoRAAttnProcessor2_0 as LoRAAttnProcessor
from tqdm.auto import tqdm
from dataset import RadarControlNetDataset
import argparse
import time
import glob


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def find_latest_checkpoint(output_dir):
    checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if not checkpoints:
        return None
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))
    return checkpoints[-1] if checkpoints else None


def compute_improved_weighted_loss(model_pred, noise, conditioning_heatmap, weight_factor=30.0, use_focal=False):
    """
    改进的加权Loss函数（与原训练保持一致），默认较温和的权重以避免过度糊化。

    Args:
        model_pred: UNet预测的噪声 [B, 4, 64, 64]
        noise: 真实噪声 [B, 4, 64, 64]
        conditioning_heatmap: 热力图条件 [B, 3, 512, 512]
        weight_factor: 目标区域的权重倍数
        use_focal: 是否使用Focal Loss机制

    Returns:
        weighted_loss: 加权后的损失
        stats: 统计信息
    """
    base_loss = F.mse_loss(model_pred.float(), noise.float(), reduction="none")  # [B, 4, 64, 64]

    # 将条件热力图下采样并生成权重
    heatmap_gray = conditioning_heatmap.mean(dim=1, keepdim=True)  # [B, 1, 512, 512]
    weight_map = F.max_pool2d(heatmap_gray, kernel_size=8, stride=8)  # [B, 1, 64, 64]

    target_mask = (weight_map > 0.1).float()  # 简单二值阈值
    weight_map = torch.where(
        target_mask > 0.5,
        torch.full_like(weight_map, float(weight_factor)),
        torch.ones_like(weight_map)
    )

    if use_focal:
        focal_weight = torch.pow(base_loss.detach() / (base_loss.detach().mean() + 1e-8), 0.5)
        weight_map = weight_map * (0.5 + 0.5 * focal_weight.mean(dim=1, keepdim=True))

    weight_map = weight_map.expand_as(base_loss)
    weighted_loss = (base_loss * weight_map).sum() / (weight_map.sum() + 1e-8)

    with torch.no_grad():
        base_loss_mean = base_loss.mean().item()
        weight_mean = weight_map.mean().item()
        target_ratio = target_mask.mean().item()
        target_loss = (base_loss * target_mask.expand_as(base_loss)).sum() / (target_mask.sum() * 4 + 1e-6)
        bg_loss = (base_loss * (1 - target_mask.expand_as(base_loss))).sum() / ((1 - target_mask).sum() * 4 + 1e-6)

    return weighted_loss, {
        'base_loss': base_loss_mean,
        'weight_mean': weight_mean,
        'target_ratio': target_ratio,
        'target_loss': target_loss.item(),
        'bg_loss': bg_loss.item(),
    }


def add_unet_lora_layers(unet: UNet2DConditionModel, rank: int = 16):
    """
    给 UNet 注入 LoRA 注意力处理器，并返回可训练的 LoRA 参数列表（兼容不含 AttnProcsLayers 的 diffusers 版本）。
    """
    lora_attn_procs = {}
    for name in unet.attn_processors.keys():
        # Self-Attn: attn1（无交叉注意力）; Cross-Attn: attn2
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim

        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks."):].split(".")[0])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks."):].split(".")[0])
            hidden_size = unet.config.block_out_channels[block_id]
        else:
            hidden_size = unet.config.block_out_channels[-1]

        lora_attn_procs[name] = LoRAAttnProcessor(
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
            rank=rank
        )

    unet.set_attn_processor(lora_attn_procs)

    # 收集可训练的 LoRA 参数
    lora_parameters = []
    for module in unet.attn_processors.values():
        if hasattr(module, 'parameters'):
            for p in module.parameters():
                p.requires_grad_(True)
                lora_parameters.append(p)

    return lora_parameters


def main(config_path, resume_from_checkpoint=None):
    config = load_config(config_path)
    proj = config["project"]
    data_cfg = config["data"]
    train_cfg = config["training"]
    ctrl_cfg = config.get("controlnet", {})

    weight_factor = float(train_cfg.get("loss_weight_factor", 30.0))
    use_focal_loss = bool(train_cfg.get("use_focal_loss", False))
    lora_rank = int(train_cfg.get("lora_rank", 16))
    heatmap_sigma = float(train_cfg.get("heatmap_sigma", 8.0))

    print(f"🎯 LoRA 微调 - 仅训练 UNet 的 LoRA 层")
    print(f"   - LoRA rank: {lora_rank}")
    print(f"   - Loss 权重因子: {weight_factor}x | Focal: {'启用' if use_focal_loss else '禁用'}")
    print(f"   - 热力图 Sigma: {heatmap_sigma}")

    accelerator = Accelerator(
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        mixed_precision=train_cfg["mixed_precision"],
    )

    # TensorBoard
    writer = None
    if accelerator.is_main_process:
        try:
            tensorboard_dir = os.path.join(proj["output_dir"], "logs")
            os.makedirs(tensorboard_dir, exist_ok=True)
            writer = SummaryWriter(log_dir=tensorboard_dir)
        except Exception:
            writer = None

    print("📥 加载 SD1.5 与已训练的 ControlNet...")
    tokenizer = CLIPTokenizer.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet")

    controlnet_path = ctrl_cfg.get("path")
    if not controlnet_path or not os.path.isdir(controlnet_path):
        raise ValueError(f"controlnet.path 未设置或不存在: {controlnet_path}")
    controlnet = ControlNetModel.from_pretrained(controlnet_path)

    noise_scheduler = DDPMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")

    # xformers（可选）
    try:
        import xformers  # noqa: F401
        unet.enable_xformers_memory_efficient_attention()
        controlnet.enable_xformers_memory_efficient_attention()
        print("✅ 启用 xformers 内存优化")
    except Exception:
        print("⚠️  xformers 未启用，使用标准注意力")

    # 设备与冻结
    vae = vae.to(accelerator.device)
    text_encoder = text_encoder.to(accelerator.device)
    unet = unet.to(accelerator.device)
    controlnet = controlnet.to(accelerator.device)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    controlnet.requires_grad_(False)

    # 注入 LoRA，仅 LoRA 训练
    lora_parameters = add_unet_lora_layers(unet, rank=lora_rank)
    unet.train()

    # 学习率与优化器
    lr_raw = train_cfg.get("learning_rate", 5e-5)
    try:
        learning_rate = float(lr_raw)
    except Exception:
        raise ValueError(f"training.learning_rate 必须是数值，当前为: {lr_raw!r}")

    optimizer = torch.optim.AdamW(lora_parameters, lr=learning_rate)

    # 数据集
    dataset = RadarControlNetDataset(
        data_dir=data_cfg["data_dir"],
        tokenizer=tokenizer,
        size=data_cfg["resolution"],
        range_max=data_cfg["range_max"],
        vel_max=data_cfg["vel_max"],
        heatmap_sigma=heatmap_sigma
    )

    train_dataloader = DataLoader(
        dataset,
        batch_size=train_cfg["train_batch_size"],
        shuffle=True,
        num_workers=train_cfg["dataloader_num_workers"]
    )

    # 训练步数与调度器
    num_update_steps_per_epoch = len(train_dataloader) // max(1, train_cfg["gradient_accumulation_steps"])
    max_train_steps = train_cfg["num_train_epochs"] * max(1, num_update_steps_per_epoch)

    print(f"📊 学习率调度器: {train_cfg['lr_scheduler']}")
    lr_scheduler = get_scheduler(
        train_cfg["lr_scheduler"],
        optimizer=optimizer,
        num_warmup_steps=train_cfg["lr_warmup_steps"] * train_cfg["gradient_accumulation_steps"],
        num_training_steps=max_train_steps * train_cfg["gradient_accumulation_steps"],
    )

    # prepare
    print("🔧 Preparing models with Accelerator...")
    unet, controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, controlnet, optimizer, train_dataloader, lr_scheduler
    )

    # 训练主循环
    total_start_time = time.time()
    global_step = 0

    print("\n" + "=" * 60)
    print("🚀 开始 LoRA 微调训练")
    print("=" * 60)

    for epoch in range(train_cfg["num_train_epochs"]):
        epoch_start_time = time.time()
        epoch_weighted_loss = 0.0
        epoch_base_loss = 0.0
        epoch_target_loss = 0.0
        epoch_bg_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(
            total=num_update_steps_per_epoch,
            desc=f"Epoch {epoch + 1}/{train_cfg['num_train_epochs']}",
            disable=not accelerator.is_local_main_process
        )

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                latents = vae.encode(batch["pixel_values"].to(dtype=vae.dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                with torch.no_grad():
                    # ControlNet 固定，仅生成附加特征
                    down_block_res_samples, mid_block_res_sample = controlnet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        controlnet_cond=batch["conditioning_pixel_values"],
                        return_dict=False,
                    )

                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                ).sample

                loss, loss_stats = compute_improved_weighted_loss(
                    model_pred,
                    noise,
                    batch["conditioning_pixel_values"],
                    weight_factor=weight_factor,
                    use_focal=use_focal_loss,
                )

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(lora_parameters, train_cfg["max_grad_norm"])
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                if accelerator.sync_gradients:
                    epoch_weighted_loss += loss.item()
                    epoch_base_loss += loss_stats['base_loss']
                    epoch_target_loss += loss_stats['target_loss']
                    epoch_bg_loss += loss_stats['bg_loss']
                    num_batches += 1
                    global_step += 1

                    avg_weighted_loss = epoch_weighted_loss / max(1, num_batches)
                    avg_base_loss = epoch_base_loss / max(1, num_batches)

                    # TB 记录
                    if writer is not None:
                        try:
                            writer.add_scalar('Loss/weighted_loss', loss.item(), global_step)
                            writer.add_scalar('Loss/base_loss', loss_stats['base_loss'], global_step)
                            writer.add_scalar('Loss/target_loss', loss_stats['target_loss'], global_step)
                            writer.add_scalar('Loss/bg_loss', loss_stats['bg_loss'], global_step)
                            writer.add_scalar('Loss/weight_mean', loss_stats['weight_mean'], global_step)
                            writer.add_scalar('Loss/target_ratio', loss_stats['target_ratio'], global_step)
                            writer.add_scalar('Training/learning_rate', optimizer.param_groups[0]['lr'], global_step)
                        except Exception:
                            pass

                    # 进度显示
                    progress_bar.set_postfix({
                        "w_loss": f"{avg_weighted_loss:.4f}",
                        "base": f"{avg_base_loss:.4f}",
                        "t_loss": f"{loss_stats['target_loss']:.4f}",
                        "bg": f"{loss_stats['bg_loss']:.4f}"
                    })
                    progress_bar.update(1)

                    # 定期保存 LoRA（注意：只保存注意力处理器参数，体积小）
                    if global_step % train_cfg["save_steps"] == 0:
                        if accelerator.is_main_process:
                            save_path = os.path.join(proj["output_dir"], f"checkpoint-{global_step}")
                            os.makedirs(save_path, exist_ok=True)
                            accelerator.unwrap_model(unet).save_attn_procs(save_path)
                            trainer_state = {
                                "epoch": epoch,
                                "global_step": global_step,
                                "resume_step": step + 1
                            }
                            with open(os.path.join(save_path, "trainer_state.yaml"), 'w') as f:
                                yaml.dump(trainer_state, f)

        progress_bar.close()

        epoch_time = time.time() - epoch_start_time
        avg_weighted_loss = epoch_weighted_loss / max(1, num_batches)
        avg_base_loss = epoch_base_loss / max(1, num_batches)
        avg_target_loss = epoch_target_loss / max(1, num_batches)
        avg_bg_loss = epoch_bg_loss / max(1, num_batches)

        # 记录 epoch 级指标
        if writer is not None:
            try:
                writer.add_scalar('Loss/weighted_loss_epoch', avg_weighted_loss, epoch + 1)
                writer.add_scalar('Loss/base_loss_epoch', avg_base_loss, epoch + 1)
                writer.add_scalar('Loss/target_loss_epoch', avg_target_loss, epoch + 1)
                writer.add_scalar('Loss/bg_loss_epoch', avg_bg_loss, epoch + 1)
                writer.add_scalar('Training/epoch_time_minutes', epoch_time / 60, epoch + 1)
                writer.add_scalar('Training/learning_rate_epoch', optimizer.param_groups[0]['lr'], epoch + 1)
            except Exception:
                pass

        print(f"\n✅ Epoch {epoch + 1} 完成:")
        print(f"   Weighted Loss: {avg_weighted_loss:.4f}")
        print(f"   Base Loss: {avg_base_loss:.4f}")
        print(f"   Target Loss: {avg_target_loss:.4f} | BG Loss: {avg_bg_loss:.4f}")
        print(f"   用时: {epoch_time/60:.1f}分钟")

        # 每若干 epoch 保存一次 LoRA
        if accelerator.is_main_process and (epoch + 1) % 5 == 0:
            save_path = os.path.join(proj["output_dir"], f"checkpoint-epoch-{epoch + 1}")
            os.makedirs(save_path, exist_ok=True)
            accelerator.unwrap_model(unet).save_attn_procs(save_path)
            trainer_state = {
                "epoch": epoch,
                "global_step": global_step,
                "resume_step": 0
            }
            with open(os.path.join(save_path, "trainer_state.yaml"), 'w') as f:
                yaml.dump(trainer_state, f)
            print(f"💾 LoRA Checkpoint saved: {save_path}")

    # 保存最终 LoRA
    if accelerator.is_main_process:
        save_path = os.path.join(proj["output_dir"], "lora")
        os.makedirs(save_path, exist_ok=True)
        accelerator.unwrap_model(unet).save_attn_procs(save_path)
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        print(f"\n🎉 LoRA 训练完成！权重保存至: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="在已训练的 ControlNet 上训练 UNet LoRA（提升画质，保持位置）")
    parser.add_argument("--config", type=str, default="config_lora.yaml", help="配置文件路径")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="指定 checkpoint 路径（可选）")
    args = parser.parse_args()

    os.makedirs("output", exist_ok=True)
    main(args.config, args.resume_from_checkpoint)


