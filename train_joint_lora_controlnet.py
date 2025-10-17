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

try:
    from diffusers.models.attention_processor import LoRAAttnProcessor
except Exception:
    from diffusers.models.attention_processor import LoRAAttnProcessor2_0 as LoRAAttnProcessor


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def compute_weighted_loss(model_pred, noise, conditioning_heatmap, weight_factor=80.0, use_focal=False):
    base_loss = F.mse_loss(model_pred.float(), noise.float(), reduction="none")  # [B, 4, 64, 64]

    heatmap_gray = conditioning_heatmap.mean(dim=1, keepdim=True)  # [B,1,512,512]
    weight_map = F.max_pool2d(heatmap_gray, kernel_size=8, stride=8)  # [B,1,64,64]
    target_mask = (weight_map > 0.1).float()
    weight_map = torch.where(target_mask > 0.5, torch.full_like(weight_map, float(weight_factor)), torch.ones_like(weight_map))

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
    lora_attn_procs = {}
    for name in unet.attn_processors.keys():
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

    lora_parameters = []
    for module in unet.attn_processors.values():
        if hasattr(module, 'parameters'):
            for p in module.parameters():
                p.requires_grad_(True)
                lora_parameters.append(p)
    return lora_parameters


def stage_train(
    accelerator,
    stage_name,
    vae,
    text_encoder,
    unet,
    controlnet,
    noise_scheduler,
    dataset,
    writer,
    train_cfg,
    weight_factor,
    use_focal_loss,
    train_controlnet: bool,
    lora_params,
    lr_cn,
    lr_lora,
):
    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg["train_batch_size"],
        shuffle=True,
        num_workers=train_cfg["dataloader_num_workers"]
    )

    params = []
    if train_controlnet:
        params.append({'params': controlnet.parameters(), 'lr': lr_cn})
    if lora_params:
        params.append({'params': lora_params, 'lr': lr_lora})
    optimizer = torch.optim.AdamW(params)

    num_update_steps_per_epoch = len(dataloader) // max(1, train_cfg["gradient_accumulation_steps"])
    max_train_steps = train_cfg["num_train_epochs"] * max(1, num_update_steps_per_epoch)
    lr_scheduler = get_scheduler(
        train_cfg["lr_scheduler"],
        optimizer=optimizer,
        num_warmup_steps=train_cfg["lr_warmup_steps"] * train_cfg["gradient_accumulation_steps"],
        num_training_steps=max_train_steps * train_cfg["gradient_accumulation_steps"],
    )

    unet, controlnet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        unet, controlnet, optimizer, dataloader, lr_scheduler
    )

    print("\n" + "="*60)
    print(f"🚀 开始训练阶段: {stage_name}")
    print("="*60)

    # 早停配置
    es_cfg = train_cfg.get("early_stopping", {})
    es_enabled = bool(es_cfg.get("enabled", False))
    es_patience = int(es_cfg.get("patience", 5))
    es_min_delta = float(es_cfg.get("min_delta", 0.0))
    best_metric = float("inf")
    bad_epochs = 0

    global_step = 0
    for epoch in range(train_cfg["num_train_epochs"]):
        epoch_weighted_loss = 0.0
        epoch_base_loss = 0.0
        epoch_target_loss = 0.0
        epoch_bg_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(total=num_update_steps_per_epoch, desc=f"{stage_name} Epoch {epoch+1}/{train_cfg['num_train_epochs']}", disable=not accelerator.is_local_main_process)

        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(unet):
                latents = vae.encode(batch["pixel_values"].to(dtype=vae.dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # ControlNet 前向（是否参与训练看 optimizer 参数）
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

                loss, loss_stats = compute_weighted_loss(
                    model_pred,
                    noise,
                    batch["conditioning_pixel_values"],
                    weight_factor=weight_factor,
                    use_focal=use_focal_loss,
                )

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    if train_controlnet:
                        accelerator.clip_grad_norm_(controlnet.parameters(), train_cfg["max_grad_norm"])
                    if lora_params:
                        accelerator.clip_grad_norm_(lora_params, train_cfg["max_grad_norm"])
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

                    if writer is not None:
                        try:
                            writer.add_scalar(f'{stage_name}/weighted_loss', loss.item(), global_step)
                            writer.add_scalar(f'{stage_name}/base_loss', loss_stats['base_loss'], global_step)
                            writer.add_scalar(f'{stage_name}/target_loss', loss_stats['target_loss'], global_step)
                            writer.add_scalar(f'{stage_name}/bg_loss', loss_stats['bg_loss'], global_step)
                            writer.add_scalar(f'{stage_name}/weight_mean', loss_stats['weight_mean'], global_step)
                            writer.add_scalar(f'{stage_name}/target_ratio', loss_stats['target_ratio'], global_step)
                            writer.add_scalar(f'{stage_name}/learning_rate', optimizer.param_groups[0]['lr'], global_step)
                        except Exception:
                            pass

                    progress_bar.set_postfix({
                        "w_loss": f"{(epoch_weighted_loss/max(1,num_batches)):.4f}",
                        "base": f"{(epoch_base_loss/max(1,num_batches)):.4f}",
                        "t": f"{loss_stats['target_loss']:.4f}",
                        "bg": f"{loss_stats['bg_loss']:.4f}",
                    })
                    progress_bar.update(1)

        progress_bar.close()

        # 计算 epoch 级指标
        avg_weighted_loss = epoch_weighted_loss / max(1, num_batches)
        avg_base_loss = epoch_base_loss / max(1, num_batches)
        avg_target_loss = epoch_target_loss / max(1, num_batches)
        avg_bg_loss = epoch_bg_loss / max(1, num_batches)

        if writer is not None:
            try:
                writer.add_scalar(f'{stage_name}/weighted_loss_epoch', avg_weighted_loss, epoch + 1)
                writer.add_scalar(f'{stage_name}/base_loss_epoch', avg_base_loss, epoch + 1)
                writer.add_scalar(f'{stage_name}/target_loss_epoch', avg_target_loss, epoch + 1)
                writer.add_scalar(f'{stage_name}/bg_loss_epoch', avg_bg_loss, epoch + 1)
            except Exception:
                pass

        # 早停判断（监控 avg_weighted_loss 降低）
        if es_enabled:
            improved = (best_metric - avg_weighted_loss) > es_min_delta
            if improved:
                best_metric = avg_weighted_loss
                bad_epochs = 0
            else:
                bad_epochs += 1
                if accelerator.is_main_process:
                    print(f"⏸️  早停观察: 未提升 {bad_epochs}/{es_patience} | 最佳 {best_metric:.6f} | 当前 {avg_weighted_loss:.6f}")
                if bad_epochs >= es_patience:
                    if accelerator.is_main_process:
                        print(f"🛑  早停触发于 {stage_name} Epoch {epoch+1} | 最佳 weighted_loss_epoch={best_metric:.6f}")
                    break


def main(config_path):
    cfg = load_config(config_path)
    proj = cfg["project"]
    data_cfg = cfg["data"]
    s1 = cfg["stage1"]
    s2 = cfg["stage2"]

    accelerator = Accelerator(
        gradient_accumulation_steps=s1["training"]["gradient_accumulation_steps"],
        mixed_precision=s1["training"]["mixed_precision"],
    )

    writer = None
    if accelerator.is_main_process:
        try:
            tensorboard_dir = os.path.join(proj["output_dir"], "logs")
            os.makedirs(tensorboard_dir, exist_ok=True)
            writer = SummaryWriter(log_dir=tensorboard_dir)
        except Exception:
            writer = None

    tokenizer = CLIPTokenizer.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet")
    controlnet = ControlNetModel.from_unet(unet)
    noise_scheduler = DDPMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")

    try:
        import xformers  # noqa: F401
        unet.enable_xformers_memory_efficient_attention()
        controlnet.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    vae = vae.to(accelerator.device)
    text_encoder = text_encoder.to(accelerator.device)
    unet = unet.to(accelerator.device)
    controlnet = controlnet.to(accelerator.device)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # 数据集（sigma 以 stage 配置为准）
    dataset = RadarControlNetDataset(
        data_dir=data_cfg["data_dir"],
        tokenizer=tokenizer,
        size=data_cfg["resolution"],
        range_max=data_cfg["range_max"],
        vel_max=data_cfg["vel_max"],
        heatmap_sigma=s1["training"].get("heatmap_sigma", 8.0)
    )

    # 阶段1：仅训练 ControlNet（定位优先）
    controlnet.requires_grad_(True)
    lora_params = []

    print("\n==== 阶段1：仅训练 ControlNet ====")
    stage_train(
        accelerator,
        stage_name="stage1",
        vae=vae,
        text_encoder=text_encoder,
        unet=unet,
        controlnet=controlnet,
        noise_scheduler=noise_scheduler,
        dataset=dataset,
        writer=writer,
        train_cfg=s1["training"],
        weight_factor=float(s1["training"].get("loss_weight_factor", 100.0)),
        use_focal_loss=bool(s1["training"].get("use_focal_loss", False)),
        train_controlnet=True,
        lora_params=lora_params,
        lr_cn=float(s1["training"].get("learning_rate", 1e-6)),
        lr_lora=0.0,
    )

    # 阶段2：轻联合（ControlNet + UNet LoRA）
    print("\n==== 阶段2：ControlNet + UNet LoRA 轻联合 ====")
    lora_params = add_unet_lora_layers(unet, rank=int(s2["training"].get("lora_rank", 16)))
    unet.train()  # 仅 LoRA 参数 requires_grad=True
    controlnet.requires_grad_(True)

    # 若阶段2的 sigma 不同，重建数据集（可选）
    if float(s2["training"].get("heatmap_sigma", s1["training"].get("heatmap_sigma", 8.0))) != float(s1["training"].get("heatmap_sigma", 8.0)):
        dataset = RadarControlNetDataset(
            data_dir=data_cfg["data_dir"],
            tokenizer=tokenizer,
            size=data_cfg["resolution"],
            range_max=data_cfg["range_max"],
            vel_max=data_cfg["vel_max"],
            heatmap_sigma=float(s2["training"].get("heatmap_sigma", 8.0))
        )

    stage_train(
        accelerator,
        stage_name="stage2",
        vae=vae,
        text_encoder=text_encoder,
        unet=unet,
        controlnet=controlnet,
        noise_scheduler=noise_scheduler,
        dataset=dataset,
        writer=writer,
        train_cfg=s2["training"],
        weight_factor=float(s2["training"].get("loss_weight_factor", 60.0)),
        use_focal_loss=bool(s2["training"].get("use_focal_loss", False)),
        train_controlnet=True,
        lora_params=lora_params,
        lr_cn=float(s2["training"].get("learning_rate", 1e-6)),
        lr_lora=float(s2["training"].get("lora_learning_rate", 5e-5)),
    )

    if accelerator.is_main_process:
        # 保存 ControlNet 与 LoRA（LoRA 通过 UNet attn_procs 导出）
        save_root = os.path.join(proj["output_dir"], "final")
        os.makedirs(save_root, exist_ok=True)
        accelerator.unwrap_model(controlnet).save_pretrained(os.path.join(save_root, "controlnet"))
        accelerator.unwrap_model(unet).save_attn_procs(os.path.join(save_root, "lora"))
        print(f"\n🎉 训练完成，已保存至: {save_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="两阶段 + 轻联合：ControlNet → ControlNet+LoRA")
    parser.add_argument("--config", type=str, default="config_joint.yaml")
    args = parser.parse_args()
    main(args.config)


