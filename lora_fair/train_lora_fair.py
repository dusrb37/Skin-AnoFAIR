#!/usr/bin/env python
# coding=utf-8
"""
Skin-AnoFAIR: LoRAfair Training
"""

import argparse
import logging
import os
import random
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from torchvision import transforms

from diffusers import (
    AutoencoderKL,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel,
    StableDiffusionPipeline
)
from diffusers.loaders import LoraLoaderMixin
from diffusers.optimization import get_scheduler
from diffusers.utils import convert_state_dict_to_diffusers

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 2: Train LoRAfair for Skin-AnoFAIR")
    
    # Model
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--output_dir", type=str, default="./models/lora_fair")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", type=int, default=512)
    
    # Dataset paths
    parser.add_argument("--clinical_train_metadata", type=str, required=True)
    parser.add_argument("--clinical_val_metadata", type=str, required=True)
    parser.add_argument("--fairface_train_metadata", type=str, required=True)
    
    # Classifier paths
    parser.add_argument("--gender_classifier_path", type=str, required=True)
    parser.add_argument("--race_classifier_path", type=str, required=True)
    parser.add_argument("--age_classifier_path", type=str, required=True)
    
    # Training parameters
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--images_per_prompt", type=int, default=16)
    parser.add_argument("--max_train_steps", type=int, default=5000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    
    # LoRA configuration
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=16)
    
    # Optimizer
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    
    # Generation
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    
    # Logging & Checkpointing
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--validation_steps", type=int, default=500)
    
    args = parser.parse_args()
    return args


class CombinedFairnessDataset(Dataset):
    """dataset"""
    def __init__(self, clinical_path, fairface_path):
        self.prompts = []
        self.metadata = []
        
        # Load clinical data
        if clinical_path and os.path.exists(clinical_path):
            with open(clinical_path, 'r') as f:
                clinical_data = json.load(f)
                for item in clinical_data:
                    age = item['age']
                    gender = item['gender']
                    disease = item['disease']
                    race = item.get('race', 'unknown')
                    
                    prompt = f"A {age} years old {gender} with {disease}"
                    self.prompts.append(prompt)
                    self.metadata.append({
                        'gender': gender,
                        'age': age,
                        'race': race,
                        'disease': disease,
                        'source': 'clinical'
                    })
                logger.info(f"Loaded {len(clinical_data)} clinical samples")
        
        # Load FairFace data
        if fairface_path and os.path.exists(fairface_path):
            with open(fairface_path, 'r') as f:
                fairface_data = json.load(f)
                for item in fairface_data:
                    age = item['age']
                    gender = item['gender']
                    race = item['race']
                    
                    prompt = f"A {age} years old {race} {gender} face"
                    self.prompts.append(prompt)
                    self.metadata.append({
                        'gender': gender,
                        'age': age,
                        'race': race,
                        'disease': None,
                        'source': 'fairface'
                    })
                logger.info(f"Loaded {len(fairface_data)} FairFace samples")
        
        logger.info(f"Total dataset size: {len(self.prompts)}")
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return {
            'prompt': self.prompts[idx],
            'metadata': self.metadata[idx]
        }


class DemographicClassifiers(nn.Module):
    """Three demographic classifiers: gender, race, age"""
    def __init__(self, gender_path, race_path, age_path, device):
        super().__init__()
        import torchvision.models as models
        
        # Gender classifier (2 classes: Female=0, Male=1)
        self.gender_model = models.mobilenet_v3_large(weights=None)
        self.gender_model.classifier[-1] = nn.Linear(
            self.gender_model.classifier[-1].in_features, 2
        )
        if os.path.exists(gender_path):
            checkpoint = torch.load(gender_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                self.gender_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.gender_model.load_state_dict(checkpoint)
            logger.info(f"Loaded gender classifier from {gender_path}")
        
        # Race classifier (5 classes: White, Black, Asian, Middle Eastern, Latino)
        self.race_model = models.mobilenet_v3_large(weights=None)
        self.race_model.classifier[-1] = nn.Linear(
            self.race_model.classifier[-1].in_features, 5
        )
        if os.path.exists(race_path):
            checkpoint = torch.load(race_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                self.race_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.race_model.load_state_dict(checkpoint)
            logger.info(f"Loaded race classifier from {race_path}")
        
        # Age classifier (2 classes: Young=0, Old=1)
        self.age_model = models.mobilenet_v3_large(weights=None)
        self.age_model.classifier[-1] = nn.Linear(
            self.age_model.classifier[-1].in_features, 2
        )
        if os.path.exists(age_path):
            checkpoint = torch.load(age_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                self.age_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.age_model.load_state_dict(checkpoint)
            logger.info(f"Loaded age classifier from {age_path}")
        
        self.to(device)
        self.eval()
        
        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False
        
        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    @torch.no_grad()
    def forward(self, images):
        """Forward pass through all three classifiers"""
        # Normalize: [-1, 1] -> [0, 1] -> ImageNet norm
        images = (images + 1) / 2
        images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
        images = (images - self.mean) / self.std
        
        # Get predictions from all three models
        gender_logits = self.gender_model(images)
        race_logits = self.race_model(images)
        age_logits = self.age_model(images)
        
        return {
            'gender': gender_logits,  # [B, 2]
            'race': race_logits,      # [B, 5]
            'age': age_logits         # [B, 2]
        }


def make_grad_hook(coef):
    """Create gradient scaling hook for efficient backprop"""
    return lambda grad: coef * grad


def sample_target_distribution():
    """Sample target distribution for distributional alignment"""
    if random.random() < 0.3:  # 30% uniform distribution
        return {
            'race': torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2]),  # 5 races uniform
            'gender': torch.tensor([0.5, 0.5]),                # 2 genders uniform
            'age': torch.tensor([0.75, 0.25])                  # 75% young, 25% old
        }
    else:  # 70% slightly random but still balanced
        # Add small noise to uniform distribution
        race_probs = torch.softmax(torch.randn(5) * 0.3 + torch.ones(5), dim=0)
        gender_probs = torch.softmax(torch.randn(2) * 0.3 + torch.ones(2), dim=0)
        # Keep age biased toward young
        age_logits = torch.tensor([1.0, -0.5])
        age_probs = torch.softmax(age_logits, dim=0)
        
        return {
            'race': race_probs,
            'gender': gender_probs,
            'age': age_probs
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
    
    if args.seed is not None:
        set_seed(args.seed)
    
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Load datasets
    train_dataset = CombinedFairnessDataset(
        args.clinical_train_metadata,
        args.fairface_train_metadata
    )
    
    val_dataset = CombinedFairnessDataset(
        args.clinical_val_metadata,
        ""  # No FairFace for validation
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    
    # Load models
    from transformers import CLIPTokenizer, CLIPTextModel
    
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer"
    )
    
    noise_scheduler = DPMSolverMultistepScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder"
    )
    
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae"
    )
    
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet"
    )
    
    # Freeze base models
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    
    # Add LoRA to UNet
    unet_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.add_adapter(unet_lora_config)
    
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    
    # Move models to device
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    
    # Initialize demographic classifiers
    classifiers = DemographicClassifiers(
        args.gender_classifier_path,
        args.race_classifier_path,
        args.age_classifier_path,
        accelerator.device
    )
    
    # Setup optimizer
    params_to_optimize = list(filter(lambda p: p.requires_grad, unet.parameters()))
    
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=1e-08,
    )
    
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
    
    if accelerator.is_main_process:
        accelerator.init_trackers("skin-anofair-lorafair", config=vars(args))
    
    # Compute gradient coefficients for denoising steps
    noise_scheduler.set_timesteps(args.num_inference_steps)
    grad_coefs = []
    for t in noise_scheduler.timesteps:
        alpha_prod_t = noise_scheduler.alphas_cumprod[t]
        alpha_prod_t_prev = noise_scheduler.alphas_cumprod[max(0, t-1)]
        grad_coef = alpha_prod_t.sqrt() * (1 - alpha_prod_t).sqrt() / (1 - alpha_prod_t_prev)
        grad_coefs.append(grad_coef.item())
    grad_coefs = np.array(grad_coefs)
    grad_coefs = grad_coefs / (np.prod(grad_coefs) ** (1/len(grad_coefs)))  # Normalize
    
    @torch.no_grad()
    def validate_distribution_alignment():
        """Validate DAE on clinical validation set"""
        unet.eval()
        
        # Fixed uniform target for validation
        val_target_dist = {
            'race': torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2]).to(accelerator.device),
            'gender': torch.tensor([0.5, 0.5]).to(accelerator.device),
            'age': torch.tensor([0.75, 0.25]).to(accelerator.device)
        }
        
        # Sample validation prompts
        num_val_samples = min(16, len(val_dataset))
        val_indices = random.sample(range(len(val_dataset)), num_val_samples)
        val_prompts = [val_dataset[i]['prompt'] for i in val_indices]
        
        # Generate images using pipeline
        pipeline = StableDiffusionPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=accelerator.unwrap_model(unet),
            scheduler=noise_scheduler,
            safety_checker=None,
            feature_extractor=None
        )
        pipeline = pipeline.to(accelerator.device)
        
        generator = torch.Generator(device=accelerator.device).manual_seed(42)
        images = pipeline(
            val_prompts,
            num_inference_steps=50,
            guidance_scale=7.5,
            generator=generator
        ).images
        
        # Convert PIL images to tensor
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        images_tensor = torch.stack([transform(img) for img in images]).to(accelerator.device)
        
        # Measure demographic distribution
        demographic_logits = classifiers(images_tensor)
        
        # Calculate Distribution Alignment Error (DAE)
        dae_total = 0
        dae_details = {}
        
        for attr in ['race', 'gender', 'age']:
            probs = F.softmax(demographic_logits[attr], dim=-1)
            measured_dist = probs.mean(dim=0)  # Batch average
            target = val_target_dist[attr]
            dae = F.l1_loss(measured_dist, target)
            dae_total += dae.item()
            dae_details[attr] = dae.item()
        
        # Cleanup
        del pipeline
        torch.cuda.empty_cache()
        unet.train()
        
        return dae_total / 3, dae_details
    
    # Training info
    logger.info("***** Running LoRAfair Training *****")
    logger.info(f"  Training dataset size = {len(train_dataset)}")
    logger.info(f"  Validation dataset size = {len(val_dataset)}")
    logger.info(f"  Max training steps = {args.max_train_steps}")
    logger.info(f"  Images per prompt = {args.images_per_prompt}")
    logger.info(f"  LoRA Rank = {args.rank}")
    
    global_step = 0
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Training Steps")
    
    unet.train()
    epoch = 0
    
    # Training loop
    while global_step < args.max_train_steps:
        epoch += 1
        for batch in train_dataloader:
            if global_step >= args.max_train_steps:
                break
            
            with accelerator.accumulate(unet):
                # Get prompts from dataset
                prompts = batch['prompt']
                
                # Expand prompts for multiple generations
                expanded_prompts = []
                for prompt in prompts:
                    expanded_prompts.extend([prompt] * args.images_per_prompt)
                
                # Sample target distribution
                target_dist = sample_target_distribution()
                for k in target_dist:
                    target_dist[k] = target_dist[k].to(accelerator.device)
                
                # Encode prompts
                text_inputs = tokenizer(
                    expanded_prompts,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_embeddings = text_encoder(
                    text_inputs.input_ids.to(accelerator.device)
                )[0]
                
                # Unconditional embeddings for CFG
                uncond_tokens = [""] * len(expanded_prompts)
                uncond_input = tokenizer(
                    uncond_tokens,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                uncond_embeddings = text_encoder(
                    uncond_input.input_ids.to(accelerator.device)
                )[0]
                
                # Initialize random noise
                batch_size = len(expanded_prompts)
                latents = torch.randn(
                    (batch_size, 4, 64, 64),
                    device=accelerator.device,
                    dtype=weight_dtype
                )
                
                # Denoising loop with gradient hooks
                noise_scheduler.set_timesteps(args.num_inference_steps)
                
                for i, t in enumerate(noise_scheduler.timesteps):
                    # Expand latents for CFG
                    latent_model_input = torch.cat([latents] * 2)
                    latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)
                    
                    # Concat embeddings for CFG
                    prompt_embeds = torch.cat([uncond_embeddings, text_embeddings])
                    
                    # UNet forward pass
                    noise_pred = unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                    ).sample
                    
                    # Classifier-free guidance
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_text - noise_pred_uncond)
                    
                    # Apply gradient hook for efficient backprop
                    hook_handle = noise_pred.register_hook(make_grad_hook(grad_coefs[i]))
                    
                    # Compute x_{t-1}
                    latents = noise_scheduler.step(noise_pred, t, latents).prev_sample
                    
                    # Remove hook to prevent memory leak
                    hook_handle.remove()
                
                # Decode latents to images
                latents = 1 / vae.config.scaling_factor * latents
                images = vae.decode(latents.to(vae.dtype)).sample.clamp(-1, 1)
                
                # Measure demographic distribution
                demographic_logits = classifiers(images)
                
                # Compute LDA loss (Equation 3)
                loss = 0
                for attr in ['race', 'gender', 'age']:
                    # Get predicted distribution (batch average)
                    probs = F.softmax(demographic_logits[attr], dim=-1)
                    predicted_dist = probs.mean(dim=0)
                    
                    # L1 loss between distributions
                    loss_attr = F.l1_loss(predicted_dist, target_dist[attr])
                    loss += loss_attr
                
                # Backward pass
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_optimize, args.max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Logging and checkpointing
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                logs = {
                    "train/loss": loss.detach().item(),
                    "train/lr": lr_scheduler.get_last_lr()[0],
                    "train/epoch": epoch
                }
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                
                # Validation
                if global_step % args.validation_steps == 0:
                    if accelerator.is_main_process:
                        val_dae, val_details = validate_distribution_alignment()
                        logger.info(f"Step {global_step} - Validation DAE: {val_dae:.4f}")
                        logger.info(f"  Race DAE: {val_details['race']:.4f}")
                        logger.info(f"  Gender DAE: {val_details['gender']:.4f}")
                        logger.info(f"  Age DAE: {val_details['age']:.4f}")
                        
                        accelerator.log({
                            "val/dae_total": val_dae,
                            "val/dae_race": val_details['race'],
                            "val/dae_gender": val_details['gender'],
                            "val/dae_age": val_details['age']
                        }, step=global_step)
                
                # Save checkpoint
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved checkpoint to {save_path}")
    
    # Save final model
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
        
        from safetensors.torch import save_file
        save_file(
            unet_lora_state_dict,
            os.path.join(args.output_dir, "lora_fair.safetensors"),
        )
        
        logger.info(f"LoRAfair training completed!")
        logger.info(f"Model saved to {args.output_dir}/lora_fair.safetensors")
    
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
