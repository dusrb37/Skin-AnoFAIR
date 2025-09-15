#!/usr/bin/env python
# coding=utf-8
"""
Skin-AnoFAIR: Stage 1 - LoRAbias Training
"""

import argparse
import logging
import math
import os
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from pathlib import Path
from tqdm.auto import tqdm
import pandas as pd
import numpy as np

import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from PIL import Image
from torchvision import transforms

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.loaders import LoraLoaderMixin
from diffusers.optimization import get_scheduler
from diffusers.utils import convert_state_dict_to_diffusers

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1: Train LoRAbias for Skin-AnoFAIR")
    
    # Model configuration
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Path to pretrained model",
    )
    
    # Dataset configuration
    parser.add_argument(
        "--train_data_dir",
        type=str,
        required=True,
        help="Directory containing training images and metadata.csv",
    )
    
    parser.add_argument(
        "--validation_data_dir",
        type=str,
        default=None,
        help="Directory containing validation images and metadata.csv",
    )
    
    # Training configuration
    parser.add_argument("--output_dir", type=str, default="./models/lora_bias")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    
    # Optimizer configuration
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--use_8bit_adam", action="store_true")
    
    # LoRA configuration
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=16)
    
    # Logging configuration
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--validation_steps", type=int, default=500)
    
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", type=str, default=None)
    
    args = parser.parse_args()
    return args


class FacialSkinDiseaseDataset(torch.utils.data.Dataset):
    """
    Dataset for facial skin disease images (6,000 images from paper)
    Structure:
    - images/: original images  
    - metadata.csv: [filename, disease, age, gender, race]
    """
    
    def __init__(self, data_dir, tokenizer, resolution=512):
        self.data_dir = Path(data_dir)
        self.resolution = resolution
        self.tokenizer = tokenizer
        
        # Load metadata
        self.metadata = pd.read_csv(self.data_dir / "metadata.csv")
        
        # Image transforms
        self.image_transforms = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
            
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        
        # Load images
        image_path = self.data_dir / "images" / row['filename']
        image = Image.open(image_path).convert("RGB")
        
        # Apply transforms
        image = self.image_transforms(image)
                
        # Create prompt (from paper Figure 3 format)
        prompt = f"A {row['age']} years old {row['race']} {row['gender']} with {row['disease']}"
        
        # Tokenize
        text_inputs = self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        
        return {
            "pixel_values": image,
            "input_ids": text_inputs.input_ids[0],
            "prompt": prompt,
        }


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=logging_dir
    )
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    
    if args.seed is not None:
        set_seed(args.seed)
    
    # Create output directory
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Load tokenizer and models (SD v1.5)
    from transformers import CLIPTokenizer, CLIPTextModel
    
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
    )
    
    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler"
    )
    
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
    )
    
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
    )
    
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
    )
    
    # Freeze base model (train LoRA)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    
    # Setup LoRA configuration
    unet_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    
    # Add LoRA adapter
    unet.add_adapter(unet_lora_config)
    
    # Move to device and dtype
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    
    # Setup optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_class = bnb.optim.AdamW8bit
        except ImportError:
            raise ImportError("Please install bitsandbytes")
    else:
        optimizer_class = torch.optim.AdamW
    
    # Only optimize LoRA parameters
    params_to_optimize = list(filter(lambda p: p.requires_grad, unet.parameters()))
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    # Create dataset and dataloader
    train_dataset = FacialSkinDiseaseDataset(
        args.train_data_dir,
        tokenizer,
        resolution=args.resolution
    )
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
    )
    
    # Validation dataloader (if provided)
    val_dataloader = None
    if args.validation_data_dir:
        val_dataset = FacialSkinDiseaseDataset(
            args.validation_data_dir,
            tokenizer,
            resolution=args.resolution
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.train_batch_size,
            shuffle=False,
            num_workers=args.dataloader_num_workers,
        )
    
    # Setup learning rate scheduler
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )
    
    # Prepare with accelerator
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )
    
    if val_dataloader is not None:
        val_dataloader = accelerator.prepare(val_dataloader)
    
    # Initialize tracker
    if accelerator.is_main_process:
        tracker_config = vars(args)
        accelerator.init_trackers("skin-anofair-bias", config=tracker_config)
    
    # Define validation functions
    def compute_validation_loss(vae, text_encoder, unet, dataloader, noise_scheduler, weight_dtype, num_batches=10):
        """Compute validation loss for monitoring"""
        val_losses = []
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_batches:
                    break
                    
                latents = vae.encode(batch["pixel_values"].to(weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
                
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                
                loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")
                val_losses.append(loss.item())
        
        return sum(val_losses) / len(val_losses) if val_losses else 0
    
    @torch.no_grad()
    def generate_validation_samples(pipeline, validation_prompts, output_dir, global_step):
        """Generate sample images for visual inspection"""
        images = []
        generator = torch.Generator(device=pipeline.device).manual_seed(42)
        
        for prompt in validation_prompts:
            image = pipeline(
                prompt,
                num_inference_steps=50,
                generator=generator,
                guidance_scale=7.5,
            ).images[0]
            images.append(image)
        
        # Save grid
        if len(images) > 0:
            grid_cols = min(len(images), 2)
            grid_rows = (len(images) + grid_cols - 1) // grid_cols
            w, h = images[0].size
            grid = Image.new('RGB', (grid_cols * w, grid_rows * h))
            
            for i, img in enumerate(images):
                grid.paste(img, ((i % grid_cols) * w, (i // grid_cols) * h))
            
            save_path = os.path.join(output_dir, f"samples/step_{global_step}.png")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            grid.save(save_path)
            return grid
        return None
    
    # Validation prompts
    VALIDATION_PROMPTS = [
        "A 30 years old asian male with psoriasis",
        "A 20 years old black female with acne",
        "A 50 years old white male with rosacea",
        "A 40 years old latino female with atopic dermatitis",
    ]
    
    # Training info
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    
    logger.info("***** Running LoRAbias Training (Stage 1) *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  LoRA Rank = {args.rank}, Alpha = {args.lora_alpha}")
    
    global_step = 0
    first_epoch = 0
    
    progress_bar = tqdm(
        range(args.max_train_steps),
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    
    # Training loop
    unet.train()
    for epoch in range(first_epoch, args.num_train_epochs):
        epoch_loss = 0  # Initialize epoch loss
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                
                # Sample noise
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                
                # Sample random timesteps
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (bsz,), device=latents.device
                )
                timesteps = timesteps.long()
                
                # Add noise to latents (forward diffusion)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Get text embeddings
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                
                # Predict noise with LoRAbias
                noise_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states,
                ).sample
                
                # Calculate loss (Equation 2 from paper)
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                
                # Standard diffusion loss
                loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")
                epoch_loss += loss.detach().item()
                
                # Backward pass
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_optimize, args.max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Update progress
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                # Logging
                logs = {
                    "train/loss": loss.detach().item(),
                    "train/lr": lr_scheduler.get_last_lr()[0],
                    "train/epoch": epoch,
                    "train/step": global_step,
                }
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                
                # Save checkpoint
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved checkpoint to {save_path}")
                
                # Validation (monitoring only)
                if args.validation_steps > 0 and global_step % args.validation_steps == 0:
                    if accelerator.is_main_process:
                        unet.eval()
                        
                        # Compute validation loss
                        if val_dataloader is not None:
                            val_loss = compute_validation_loss(
                                vae, text_encoder, unet, val_dataloader,
                                noise_scheduler, weight_dtype
                            )
                            logger.info(f"Step {global_step} - Val loss: {val_loss:.4f}")
                            accelerator.log({"val/loss": val_loss}, step=global_step)
                        
                        # Generate samples (less frequent)
                        if global_step % (args.validation_steps * 4) == 0:
                            try:
                                pipeline = StableDiffusionPipeline(
                                    vae=vae,
                                    text_encoder=text_encoder,
                                    tokenizer=tokenizer,
                                    unet=accelerator.unwrap_model(unet),
                                    scheduler=noise_scheduler,
                                    safety_checker=None,
                                    feature_extractor=None,
                                )
                                pipeline = pipeline.to(accelerator.device)
                                
                                sample_grid = generate_validation_samples(
                                    pipeline, 
                                    VALIDATION_PROMPTS,
                                    args.output_dir,
                                    global_step
                                )
                                logger.info(f"Generated validation samples at step {global_step}")
                                
                                del pipeline
                                torch.cuda.empty_cache()
                                
                            except Exception as e:
                                logger.warning(f"Failed to generate samples: {e}")
                        
                        unet.train()
                
                if global_step >= args.max_train_steps:
                    break
        
        # End of epoch logging
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        logger.info(f"Epoch {epoch} completed. Avg loss: {avg_epoch_loss:.4f}")
        
        if global_step >= args.max_train_steps:
            break
    
    # Save final LoRAbias weights
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        unet = unet.to(torch.float32)
        unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
        
        LoraLoaderMixin.save_lora_weights(
            save_directory=args.output_dir,
            unet_lora_layers=unet_lora_state_dict,
            text_encoder_lora_layers=None,
        )
        
        # Save as safetensors format
        from safetensors.torch import save_file
        save_file(
            unet_lora_state_dict,
            os.path.join(args.output_dir, "lora_bias.safetensors"),
        )
        
        # Save metadata
        metadata = {
            "base_model": args.pretrained_model_name_or_path,
            "rank": args.rank,
            "alpha": args.lora_alpha,
            "target_modules": ["to_k", "to_q", "to_v", "to_out.0"],
            "training_steps": global_step,
            "epochs": epoch + 1,
        }
        
        import json
        with open(os.path.join(args.output_dir, "adapter_config.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"LoRAbias saved to {args.output_dir}")
        
        if args.push_to_hub:
            upload_folder(
                repo_id=args.hub_model_id or "skin-anofair-lora-bias",
                folder_path=args.output_dir,
                commit_message="LoRAbias (Stage 1) training complete",
            )
    
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)