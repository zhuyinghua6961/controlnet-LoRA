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

def compute_improved_weighted_loss(model_pred, noise, conditioning_heatmap, weight_factor=50.0, use_focal=False):
    """
    改进的加权Loss函数 - 修复3个关键问题
    
    核心改进：
    1. 使用Max Pooling下采样（保持峰值强度）
    2. 二值化权重（目标区域极高权重，背景标准权重）
    3. 可选：Focal Loss机制（关注难学习的区域）
    
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
    # 1. 基础MSE损失（逐像素）
    base_loss = F.mse_loss(model_pred.float(), noise.float(), reduction="none")  # [B, 4, 64, 64]
    
    # 2. 生成权重图 - 使用Max Pooling保持峰值
    heatmap_gray = conditioning_heatmap.mean(dim=1, keepdim=True)  # [B, 1, 512, 512]
    
    # 🔴 关键改进1：使用Max Pooling代替Bilinear（保持峰值强度）
    # 512 -> 64 需要下采样8倍，使用kernel_size=8, stride=8
    weight_map = F.max_pool2d(heatmap_gray, kernel_size=8, stride=8)  # [B, 1, 64, 64]
    
    # 🔴 关键改进2：二值化权重（目标/背景明确区分）
    # 阈值选择：热图值>0.1认为是目标区域
    target_mask = (weight_map > 0.1).float()  # [B, 1, 64, 64]
    
    # 目标区域：weight_factor倍权重；背景区域：1倍权重
    weight_map = torch.where(
        target_mask > 0.5,
        torch.full_like(weight_map, float(weight_factor)),  # 目标区域
        torch.ones_like(weight_map)  # 背景区域
    )
    
    # 🔴 关键改进3（可选）：Focal Loss - 给难学习的样本更高权重
    if use_focal:
        # Focal loss: 给高loss的区域更高权重（模型还没学好的地方）
        focal_weight = torch.pow(base_loss.detach() / base_loss.detach().mean(), 0.5)
        weight_map = weight_map * (0.5 + 0.5 * focal_weight.mean(dim=1, keepdim=True))
    
    # 4. 应用权重（扩展到4个通道）
    weight_map = weight_map.expand_as(base_loss)  # [B, 4, 64, 64]
    weighted_loss = (base_loss * weight_map).sum() / weight_map.sum()  # 加权平均
    
    # 5. 计算统计信息（用于监控）
    with torch.no_grad():
        base_loss_mean = base_loss.mean().item()
        weight_mean = weight_map.mean().item()
        target_ratio = target_mask.mean().item()  # 目标区域占比
        effective_weight = weight_mean  # 实际平均权重
        
        # 计算目标区域和背景区域的loss
        target_loss = (base_loss * target_mask.expand_as(base_loss)).sum() / (target_mask.sum() * 4 + 1e-6)
        bg_loss = (base_loss * (1 - target_mask.expand_as(base_loss))).sum() / ((1 - target_mask).sum() * 4 + 1e-6)
    
    return weighted_loss, {
        'base_loss': base_loss_mean,
        'weight_mean': effective_weight,
        'target_ratio': target_ratio,
        'target_loss': target_loss.item(),
        'bg_loss': bg_loss.item(),
    }

def main(config_path, resume_from_checkpoint=None):
    config = load_config(config_path)
    proj = config["project"]
    data_cfg = config["data"]
    train_cfg = config["training"]
    
    # 从配置中读取参数
    weight_factor = train_cfg.get("loss_weight_factor", 50.0)
    use_focal_loss = train_cfg.get("use_focal_loss", False)
    print(f"🎯 使用改进的加权Loss")
    print(f"   - 权重因子: {weight_factor}x")
    print(f"   - Max Pooling下采样（保持峰值）")
    print(f"   - 二值化权重（目标{weight_factor}x，背景1x）")
    print(f"   - Focal Loss: {'启用' if use_focal_loss else '禁用'}")

    accelerator = Accelerator(
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        mixed_precision=train_cfg["mixed_precision"],
    )
    
    # 初始化TensorBoard
    writer = None
    if accelerator.is_main_process:
        try:
            tensorboard_dir = os.path.join(proj["output_dir"], "logs")
            os.makedirs(tensorboard_dir, exist_ok=True)
            writer = SummaryWriter(log_dir=tensorboard_dir)
        except Exception:
            writer = None

    print("📥 加载预训练模型...")
    tokenizer = CLIPTokenizer.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet")
    controlnet = ControlNetModel.from_unet(unet)
    noise_scheduler = DDPMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")

    # 启用xformers（如果可用）
    try:
        import xformers
        unet.enable_xformers_memory_efficient_attention()
        controlnet.enable_xformers_memory_efficient_attention()
        print("✅ 启用 xformers 内存优化")
    except ImportError:
        print("⚠️  xformers 未安装，使用标准注意力机制")

    vae = vae.to(accelerator.device)
    text_encoder = text_encoder.to(accelerator.device)
    unet = unet.to(accelerator.device)

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    controlnet.train()

    # 兼容字符串或数字形式的学习率配置（例如 "1e-5" 或 1e-5）
    lr_raw = train_cfg.get("learning_rate", train_cfg.get("lr", 1e-5))
    try:
        learning_rate = float(lr_raw)
    except Exception:
        raise ValueError(f"training.learning_rate 必须是数值，当前为: {lr_raw!r}")

    optimizer = torch.optim.AdamW(controlnet.parameters(), lr=learning_rate)

    # 从配置读取sigma参数
    heatmap_sigma = train_cfg.get("heatmap_sigma", 15.0)
    print(f"🎯 热力图Sigma: {heatmap_sigma}")
    
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

    starting_epoch = 0
    global_step = 0
    resume_step = 0

    # 计算训练步数
    num_update_steps_per_epoch = len(train_dataloader) // train_cfg["gradient_accumulation_steps"]
    max_train_steps = train_cfg["num_train_epochs"] * num_update_steps_per_epoch

    # 创建学习率调度器
    print(f"📊 学习率调度器: {train_cfg['lr_scheduler']}")
    lr_scheduler = get_scheduler(
        train_cfg["lr_scheduler"],
        optimizer=optimizer,
        num_warmup_steps=train_cfg["lr_warmup_steps"] * train_cfg["gradient_accumulation_steps"],
        num_training_steps=max_train_steps * train_cfg["gradient_accumulation_steps"],
    )

    # 检查是否需要恢复训练
    if resume_from_checkpoint == "none":
        print("🆕 强制从头开始训练（--from_scratch）")
        resume_from_checkpoint = None
    elif resume_from_checkpoint is None:
        resume_from_checkpoint = find_latest_checkpoint(proj["output_dir"])
        if resume_from_checkpoint:
            print(f"🔍 自动发现最新 checkpoint: {resume_from_checkpoint}")
    else:
        print(f"📌 使用指定 checkpoint: {resume_from_checkpoint}")

    # Prepare所有组件
    print("🔧 Preparing models with Accelerator...")
    controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        controlnet, optimizer, train_dataloader, lr_scheduler
    )

    # 如果有checkpoint，加载模型权重和训练状态
    if resume_from_checkpoint:
        print(f"🔄 从 checkpoint 恢复: {resume_from_checkpoint}")
        
        controlnet_state = ControlNetModel.from_pretrained(resume_from_checkpoint)
        accelerator.unwrap_model(controlnet).load_state_dict(controlnet_state.state_dict())
        
        trainer_state_path = os.path.join(resume_from_checkpoint, "trainer_state.yaml")
        if os.path.exists(trainer_state_path):
            with open(trainer_state_path, 'r') as f:
                trainer_state = yaml.safe_load(f)
            starting_epoch = trainer_state.get("epoch", 0)
            global_step = trainer_state.get("global_step", 0)
            resume_step = trainer_state.get("resume_step", 0)
            
            if resume_step == 0:
                starting_epoch += 1
                print(f"✅ 恢复状态: 已完成 {starting_epoch} 个epoch，从 Epoch {starting_epoch + 1} 继续训练")
            else:
                print(f"✅ 恢复状态: 从 Epoch {starting_epoch + 1} 第 {resume_step} 步继续训练")
        else:
            print(f"⚠️  未找到训练状态文件，从 Epoch 1 开始")
    else:
        print("🆕 从头开始训练")

    num_update_steps_per_epoch = len(train_dataloader) // train_cfg["gradient_accumulation_steps"]

    total_start_time = time.time()
    
    print("\n" + "="*60)
    print("🚀 开始训练")
    print("="*60)
    
    for epoch in range(starting_epoch, train_cfg["num_train_epochs"]):
        epoch_start_time = time.time()
        epoch_weighted_loss = 0.0
        epoch_base_loss = 0.0
        epoch_target_loss = 0.0
        epoch_bg_loss = 0.0
        num_batches = 0
        
        if resume_step > 0:
            train_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
            resume_step = 0

        progress_bar = tqdm(
            total=num_update_steps_per_epoch,
            desc=f"Epoch {epoch + 1}/{train_cfg['num_train_epochs']}",
            disable=not accelerator.is_local_main_process
        )
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(controlnet):
                latents = vae.encode(batch["pixel_values"].to(dtype=vae.dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

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

                # 🔴 使用改进的加权Loss函数
                loss, loss_stats = compute_improved_weighted_loss(
                    model_pred, 
                    noise, 
                    batch["conditioning_pixel_values"],
                    weight_factor=weight_factor,
                    use_focal=use_focal_loss
                )

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(controlnet.parameters(), train_cfg["max_grad_norm"])
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
                    
                    avg_weighted_loss = epoch_weighted_loss / num_batches
                    avg_base_loss = epoch_base_loss / num_batches
                    
                    # 记录到TensorBoard
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
                    
                    # 显示训练进度
                    progress_bar.set_postfix({
                        "w_loss": f"{avg_weighted_loss:.4f}",
                        "base": f"{avg_base_loss:.4f}",
                        "t_loss": f"{loss_stats['target_loss']:.4f}",
                        "bg": f"{loss_stats['bg_loss']:.4f}"
                    })
                    progress_bar.update(1)
                    
                    # 定期保存checkpoint
                    if global_step % train_cfg["save_steps"] == 0:
                        if accelerator.is_main_process:
                            save_path = os.path.join(proj["output_dir"], f"checkpoint-{global_step}")
                            os.makedirs(save_path, exist_ok=True)
                            accelerator.unwrap_model(controlnet).save_pretrained(save_path)
                            trainer_state = {
                                "epoch": epoch,
                                "global_step": global_step,
                                "resume_step": step + 1
                            }
                            with open(os.path.join(save_path, "trainer_state.yaml"), 'w') as f:
                                yaml.dump(trainer_state, f)

        progress_bar.close()
        
        epoch_time = time.time() - epoch_start_time
        avg_weighted_loss = epoch_weighted_loss / num_batches if num_batches > 0 else 0
        avg_base_loss = epoch_base_loss / num_batches if num_batches > 0 else 0
        avg_target_loss = epoch_target_loss / num_batches if num_batches > 0 else 0
        avg_bg_loss = epoch_bg_loss / num_batches if num_batches > 0 else 0
        
        elapsed_total = time.time() - total_start_time
        epochs_done = epoch + 1 - starting_epoch
        epochs_left = train_cfg["num_train_epochs"] - (epoch + 1)
        avg_epoch_time = elapsed_total / epochs_done if epochs_done > 0 else 0
        total_eta = avg_epoch_time * epochs_left
        
        # 记录epoch级别的指标
        if writer is not None:
            try:
                writer.add_scalar('Loss/weighted_loss_epoch', avg_weighted_loss, epoch + 1)
                writer.add_scalar('Loss/base_loss_epoch', avg_base_loss, epoch + 1)
                writer.add_scalar('Loss/target_loss_epoch', avg_target_loss, epoch + 1)
                writer.add_scalar('Loss/bg_loss_epoch', avg_bg_loss, epoch + 1)
                writer.add_scalar('Training/epoch_time_minutes', epoch_time / 60, epoch + 1)
                writer.add_scalar('Training/samples_per_second', len(dataset) / epoch_time, epoch + 1)
                writer.add_scalar('Training/learning_rate_epoch', optimizer.param_groups[0]['lr'], epoch + 1)
            except Exception:
                pass
        
        print(f"\n✅ Epoch {epoch + 1} 完成:")
        print(f"   Weighted Loss: {avg_weighted_loss:.4f}")
        print(f"   Base Loss: {avg_base_loss:.4f}")
        print(f"   Target Loss: {avg_target_loss:.4f} | BG Loss: {avg_bg_loss:.4f}")
        print(f"   用时: {epoch_time/60:.1f}分钟 | 总ETA: {total_eta/60:.1f}分钟")
        
        # 每5个epoch保存checkpoint
        if accelerator.is_main_process and (epoch + 1) % 5 == 0:
            save_path = os.path.join(proj["output_dir"], f"checkpoint-epoch-{epoch + 1}")
            os.makedirs(save_path, exist_ok=True)
            accelerator.unwrap_model(controlnet).save_pretrained(save_path)
            trainer_state = {
                "epoch": epoch,
                "global_step": global_step,
                "resume_step": 0
            }
            with open(os.path.join(save_path, "trainer_state.yaml"), 'w') as f:
                yaml.dump(trainer_state, f)
            print(f"💾 Checkpoint saved: {save_path}")

    # 保存最终模型
    if accelerator.is_main_process:
        save_path = os.path.join(proj["output_dir"], "controlnet")
        accelerator.unwrap_model(controlnet).save_pretrained(save_path)
        
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        
        print(f"\n🎉 训练完成！模型保存至: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="雷达 RD 图 ControlNet 训练（改进Loss版本）")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="配置文件路径")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="指定 checkpoint 路径")
    parser.add_argument("--from_scratch", action="store_true",
                        help="强制从头开始训练")
    args = parser.parse_args()

    os.makedirs("output", exist_ok=True)
    
    if args.from_scratch:
        print("⚡ 强制从头开始训练模式")
        resume_checkpoint = "none"
    else:
        resume_checkpoint = args.resume_from_checkpoint
    
    main(args.config, resume_checkpoint)


