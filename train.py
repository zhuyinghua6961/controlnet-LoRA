import os
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from diffusers import ControlNetModel, AutoencoderKL, UNet2DConditionModel
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

def main(config_path, resume_from_checkpoint=None):
    config = load_config(config_path)
    proj = config["project"]
    data_cfg = config["data"]
    train_cfg = config["training"]

    accelerator = Accelerator(
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        mixed_precision=train_cfg["mixed_precision"],
    )

    tokenizer = CLIPTokenizer.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet")
    controlnet = ControlNetModel.from_unet(unet)

    try:
        import xformers
        unet.enable_xformers_memory_efficient_attention()
        controlnet.enable_xformers_memory_efficient_attention()
    except ImportError:
        pass

    vae = vae.to(accelerator.device)
    text_encoder = text_encoder.to(accelerator.device)
    unet = unet.to(accelerator.device)

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    controlnet.train()

    optimizer = torch.optim.AdamW(controlnet.parameters(), lr=train_cfg["learning_rate"])

    dataset = RadarControlNetDataset(
        data_dir=data_cfg["data_dir"],
        tokenizer=tokenizer,
        size=data_cfg["resolution"],
        range_max=data_cfg["range_max"],
        vel_max=data_cfg["vel_max"]
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

    if resume_from_checkpoint is None:
        resume_from_checkpoint = find_latest_checkpoint(proj["output_dir"])

    if resume_from_checkpoint:
        print(f"🔄 从 checkpoint 恢复: {resume_from_checkpoint}")
        controlnet = ControlNetModel.from_pretrained(resume_from_checkpoint)
        controlnet, optimizer, train_dataloader = accelerator.prepare(
            controlnet, optimizer, train_dataloader
        )
        
        optimizer_path = os.path.join(resume_from_checkpoint, "optimizer.bin")
        scheduler_path = os.path.join(resume_from_checkpoint, "scheduler.bin")
        trainer_state_path = os.path.join(resume_from_checkpoint, "trainer_state.yaml")
        
        if os.path.exists(optimizer_path):
            optimizer.load_state_dict(torch.load(optimizer_path, map_location="cpu"))
        if os.path.exists(scheduler_path):
            lr_scheduler = torch.load(scheduler_path)
        else:
            num_update_steps_per_epoch = len(train_dataloader) // train_cfg["gradient_accumulation_steps"]
            max_train_steps = train_cfg["num_train_epochs"] * num_update_steps_per_epoch
            lr_scheduler = get_scheduler(
                train_cfg["lr_scheduler"],
                optimizer=optimizer,
                num_warmup_steps=train_cfg["lr_warmup_steps"] * train_cfg["gradient_accumulation_steps"],
                num_training_steps=max_train_steps * train_cfg["gradient_accumulation_steps"],
            )
        if os.path.exists(trainer_state_path):
            with open(trainer_state_path, 'r') as f:
                trainer_state = yaml.safe_load(f)
            starting_epoch = trainer_state.get("epoch", 0)
            global_step = trainer_state.get("global_step", 0)
            resume_step = trainer_state.get("resume_step", 0)
        
        print(f"✅ 恢复状态: epoch={starting_epoch}, global_step={global_step}")
    else:
        print("🆕 从头开始训练")
        num_update_steps_per_epoch = len(train_dataloader) // train_cfg["gradient_accumulation_steps"]
        max_train_steps = train_cfg["num_train_epochs"] * num_update_steps_per_epoch
        lr_scheduler = get_scheduler(
            train_cfg["lr_scheduler"],
            optimizer=optimizer,
            num_warmup_steps=train_cfg["lr_warmup_steps"] * train_cfg["gradient_accumulation_steps"],
            num_training_steps=max_train_steps * train_cfg["gradient_accumulation_steps"],
        )
        controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            controlnet, optimizer, train_dataloader, lr_scheduler
        )

    num_update_steps_per_epoch = len(train_dataloader) // train_cfg["gradient_accumulation_steps"]
    total_steps = train_cfg["num_train_epochs"] * num_update_steps_per_epoch

    total_start_time = time.time()
    
    for epoch in range(starting_epoch, train_cfg["num_train_epochs"]):
        epoch_start_time = time.time()
        epoch_loss = 0.0
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
                timesteps = torch.randint(0, 1000, (bsz,), device=latents.device).long()
                noisy_latents = noise * timesteps.view(-1, 1, 1, 1) / 1000 + latents * (1 - timesteps.view(-1, 1, 1, 1) / 1000)

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

                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(controlnet.parameters(), train_cfg["max_grad_norm"])
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                if accelerator.sync_gradients:
                    epoch_loss += loss.item()
                    num_batches += 1
                    global_step += 1
                    
                    avg_loss = epoch_loss / num_batches
                    progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})
                    progress_bar.update(1)
                    
                    if global_step % train_cfg["save_steps"] == 0:
                        if accelerator.is_main_process:
                            save_path = os.path.join(proj["output_dir"], f"checkpoint-{global_step}")
                            os.makedirs(save_path, exist_ok=True)
                            accelerator.unwrap_model(controlnet).save_pretrained(save_path)
                            torch.save(optimizer.state_dict(), os.path.join(save_path, "optimizer.bin"))
                            torch.save(lr_scheduler, os.path.join(save_path, "scheduler.bin"))
                            trainer_state = {
                                "epoch": epoch,
                                "global_step": global_step,
                                "resume_step": step + 1
                            }
                            with open(os.path.join(save_path, "trainer_state.yaml"), 'w') as f:
                                yaml.dump(trainer_state, f)

        progress_bar.close()
        
        epoch_time = time.time() - epoch_start_time
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
        
        elapsed_total = time.time() - total_start_time
        epochs_done = epoch + 1 - starting_epoch
        epochs_left = train_cfg["num_train_epochs"] - (epoch + 1)
        avg_epoch_time = elapsed_total / epochs_done if epochs_done > 0 else 0
        total_eta = avg_epoch_time * epochs_left
        epoch_eta = avg_epoch_time
        
        print(f"✅ Epoch {epoch + 1} finished | "
              f"Avg Loss: {avg_epoch_loss:.4f} | "
              f"Time: {epoch_time/60:.1f}m | "
              f"Epoch ETA: {epoch_eta/60:.1f}m | "
              f"Total ETA: {total_eta/60:.1f}m")

    if accelerator.is_main_process:
        save_path = os.path.join(proj["output_dir"], "controlnet")
        accelerator.unwrap_model(controlnet).save_pretrained(save_path)
        print(f"🎉 训练完成！模型保存至: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="指定 checkpoint 路径")
    args = parser.parse_args()

    os.makedirs("output", exist_ok=True)
    main(args.config, args.resume_from_checkpoint)