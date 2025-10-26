#!/usr/bin/env python
# coding=utf-8
"""
Skin-AnoFAIR: LoRASelf-PO Training
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
import lpips

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from torchvision import transforms
from PIL import Image

from diffusers import (
    AutoencoderKL,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel,
    StableDiffusionPipeline
)
from diffusers.loaders import LoraLoaderMixin
from diffusers.optimization import get_scheduler
from diffusers.utils import convert_state_dict_to_diffusers
from diffusers import StableDiffusionImg2ImgPipeline

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 3: Train LoRASelf-PO for Skin-AnoFAIR")
    
    # Model configuration
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--output_dir", type=str, default="./models/lora_self_po")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", type=int, default=512)
    
    # Dataset paths (clinical only)
    parser.add_argument("--clinical_train_metadata", type=str, required=True, 
                       help="Path to clinical training metadata JSON")
    parser.add_argument("--clinical_val_metadata", type=str, required=True,
                       help="Path to clinical validation metadata JSON")
    parser.add_argument("--clinical_train_images", type=str, required=True,
                   help="Path to clinical training images directory")
    parser.add_argument("--clinical_val_images", type=str, required=True,
                   help="Path to clinical validation images directory")

    # Pre-trained classifier paths
    parser.add_argument("--disease_classifier_path", type=str, required=True,
                       help="Path to disease classifier (6 classes)")
    parser.add_argument("--gender_classifier_path", type=str, required=True,
                       help="Path to gender classifier (2 classes)")
    parser.add_argument("--race_classifier_path", type=str, required=True,
                       help="Path to race classifier (5 classes)")
    parser.add_argument("--age_classifier_path", type=str, required=True,
                       help="Path to age classifier (2 classes)")
    
    # Preference generation parameters
    parser.add_argument("--num_candidates", type=int, default=8, 
                       help="k: number of candidates per prompt")
    parser.add_argument("--num_high_quality", type=int, default=4, 
                       help="m: size of high-quality subset")
    
    # Training parameters
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=3000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    
    # LoRA configuration
    parser.add_argument("--rank", type=int, default=32, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    
    # DPO parameters
    parser.add_argument("--beta", type=float, default=0.3, help="DPO beta parameter")
    
    # Optimizer configuration
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=100)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    
    # Generation parameters
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    
    # Logging and checkpointing
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--validation_steps", type=int, default=500)
    parser.add_argument("--num_validation_prompts", type=int, default=10,
                       help="Number of prompts to use for validation")
    
    args = parser.parse_args()
    return args


class ClinicalDataset(Dataset):
    """Clinical dataset for self-supervised preference learning"""
    def __init__(self, metadata_path, image_dir, resolution=512):
        self.data = []
        self.image_dir = image_dir
        self.resolution = resolution
        
        # Image transform
        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                data = json.load(f)
                for item in data:
                    image_path = os.path.join(image_dir, item['image_path'])
                    if not os.path.exists(image_path):
                        logger.warning(f"Image not found: {image_path}")
                        continue
                    
                    self.data.append({
                        'image_path': image_path,
                        'prompt': f"A {item['age']} years old {item['gender']} with {item['disease']}",
                        'metadata': {
                            'gender': item['gender'],
                            'age': item['age'],
                            'race': item.get('race', 'unknown'),
                            'disease': item['disease'],
                            'image_id': item.get('image_id', f"clinical_{len(self.data):05d}")
                        }
                    })
                logger.info(f"Loaded {len(self.data)} clinical images from {metadata_path}")
        else:
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load and transform image
        try:
            image = Image.open(item['image_path']).convert('RGB')
            image_tensor = self.transform(image)
        except Exception as e:
            logger.error(f"Error loading image {item['image_path']}: {e}")
            image_tensor = torch.zeros(3, self.resolution, self.resolution)
        
        return {
            'image': image_tensor,
            'prompt': item['prompt'],
            'metadata': item['metadata']
        }

class QualityAssessors(nn.Module):
    """All classifiers for quality assessment"""
    def __init__(self, disease_path, gender_path, race_path, age_path, device):
        super().__init__()
        import torchvision.models as models
        
        # Disease classifier (6 classes)
        self.disease_model = models.mobilenet_v3_large(weights=None)
        self.disease_model.classifier[-1] = nn.Linear(
            self.disease_model.classifier[-1].in_features, 6
        )
        self._load_model(self.disease_model, disease_path, "disease", device)
        
        # Gender classifier (2 classes)
        self.gender_model = models.mobilenet_v3_large(weights=None)
        self.gender_model.classifier[-1] = nn.Linear(
            self.gender_model.classifier[-1].in_features, 2
        )
        self._load_model(self.gender_model, gender_path, "gender", device)
        
        # Race classifier (5 classes)
        self.race_model = models.mobilenet_v3_large(weights=None)
        self.race_model.classifier[-1] = nn.Linear(
            self.race_model.classifier[-1].in_features, 5
        )
        self._load_model(self.race_model, race_path, "race", device)
        
        # Age classifier (2 classes)
        self.age_model = models.mobilenet_v3_large(weights=None)
        self.age_model.classifier[-1] = nn.Linear(
            self.age_model.classifier[-1].in_features, 2
        )
        self._load_model(self.age_model, age_path, "age", device)
        
        # DINOv2 for visual quality assessment
        self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        logger.info("Loaded DINOv2 for visual quality assessment")
        
        self.to(device)
        self.eval()
        
        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False
        
        # ImageNet normalization constants
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def _load_model(self, model, path, name, device):
        """Helper to load model weights"""
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            logger.info(f"Loaded {name} classifier from {path}")
        else:
            logger.warning(f"{name} classifier not found at {path}")
    
    @torch.no_grad()
    def forward(self, images):
        """Compute all quality scores"""
        # Normalize for classifiers: [-1, 1] -> [0, 1]
        images_norm = (images + 1) / 2
        images_224 = F.interpolate(images_norm, size=(224, 224), mode='bilinear', align_corners=False)
        images_norm = (images_224 - self.mean) / self.std
        
        # Get classifier predictions
        disease_logits = self.disease_model(images_norm)
        gender_logits = self.gender_model(images_norm)
        race_logits = self.race_model(images_norm)
        age_logits = self.age_model(images_norm)
        
        # Get DINOv2 features
        dino_features = self.dino(images_224)
        
        return {
            'disease': disease_logits,
            'gender': gender_logits,
            'race': race_logits,
            'age': age_logits,
            'dino_features': dino_features
        }
    
    def compute_quality_scores(self, images, metadata):
        """Compute unified quality score for each image"""
        outputs = self.forward(images)
        batch_size = images.shape[0]
        scores = []
        
        for i in range(batch_size):
            score = 0.0
            
            # Disease accuracy (weight: 0.4)
            if 'disease' in metadata:
                disease_pred = outputs['disease'][i].argmax().item()
                disease_gt = metadata['disease']
                disease_map = {
                    'psoriasis': 0, 'acne': 1, 'rosacea': 2,
                    'atopic_dermatitis': 3, 'seborrheic_dermatitis': 4, 'normal': 5
                }
                if disease_gt in disease_map:
                    score += 0.4 * (disease_pred == disease_map[disease_gt])
            
            # Demographic alignment (weight: 0.3)
            demo_score = 0.0
            demo_count = 0
            
            for attr in ['gender', 'race', 'age']:
                if attr in metadata:
                    conf = F.softmax(outputs[attr][i], dim=0).max().item()
                    demo_score += conf
                    demo_count += 1
            
            if demo_count > 0:
                score += 0.3 * (demo_score / demo_count)
            
            # Visual quality from DINOv2 (weight: 0.3)
            dino_norm = outputs['dino_features'][i].norm().item()
            score += 0.3 * min(dino_norm / 100.0, 1.0)
            
            scores.append(score)
        
        return torch.tensor(scores, device=images.device)


class ConsistencyValidator:
    """Validate generation consistency using LPIPS"""
    def __init__(self, device):
        self.device = device
        self.lpips_fn = lpips.LPIPS(net='alex').to(device)
        self.lpips_fn.eval()
        
        for param in self.lpips_fn.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def validate_consistency(self, pipeline, val_prompts, num_samples=3):
        """
        Validate generation consistency using LPIPS
        Lower LPIPS score = Better consistency
        """
        consistency_scores = []
        
        for prompt in tqdm(val_prompts, desc="Validating consistency"):
            images = []
            
            # Generate multiple images with same prompt
            for i in range(num_samples):
                generator = torch.Generator(device=self.device).manual_seed(42 + i)
                image = pipeline(
                    prompt,
                    num_inference_steps=50,
                    guidance_scale=7.5,
                    generator=generator
                ).images[0]
                
                # Convert to tensor [-1, 1]
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5])
                ])
                images.append(transform(image).unsqueeze(0).to(self.device))
            
            # Compute pairwise LPIPS scores
            lpips_scores = []
            for i in range(len(images)):
                for j in range(i+1, len(images)):
                    score = self.lpips_fn(images[i], images[j]).item()
                    lpips_scores.append(score)
            
            consistency_scores.append(np.mean(lpips_scores))
        
        return np.mean(consistency_scores), np.std(consistency_scores)


def generate_preference_pairs(pipeline, dataset, assessors, args, device):
    """Generate preference pairs using self-supervised method"""
    preference_data = []
    
    for idx in tqdm(range(len(dataset)), desc="Generating preference pairs"):
        data = dataset[idx]
        
        image = data['image']  # [C, H, W] tensor
        prompt = data['prompt']
        metadata = data['metadata']
        
        # Convert tensor to PIL Image for pipeline
        image_pil = transforms.ToPILImage()(image.cpu() * 0.5 + 0.5) 
        
        # Step 1: Generate k candidates with different seeds
        candidates = []
        for seed_idx in range(args.num_candidates):
            generator = torch.Generator(device=device).manual_seed(args.seed + seed_idx + idx * 1000)
            
            output = pipeline(
                prompt=prompt,
                image=image_pil,
                strength=0.7,  # Denoising strength
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=generator
            ).images[0]
            
            # Convert to tensor
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
            candidates.append(transform(output).unsqueeze(0).to(device))
        
        candidates = torch.cat(candidates, dim=0)
        
        # Step 2: Quality assessment
        quality_scores = assessors.compute_quality_scores(candidates, metadata)
        
        # Step 3: Select top-m high quality samples
        top_indices = quality_scores.topk(args.num_high_quality).indices
        high_quality_samples = candidates[top_indices]
        
        # Get DINOv2 features for centroid computation
        outputs = assessors(high_quality_samples)
        dino_features = outputs['dino_features']
        
        # Compute centroid in feature space
        centroid = dino_features.mean(dim=0, keepdim=True)
        
        # Find winner (closest to centroid)
        distances = torch.cdist(dino_features, centroid, p=2).squeeze()
        winner_idx = distances.argmin().item()
        
        # Create preference pairs (1 winner vs m-1 losers)
        for i in range(args.num_high_quality):
            if i != winner_idx:
                # Convert local indices to global indices
                real_winner_idx = top_indices[winner_idx].item()
                real_loser_idx = top_indices[i].item()
                
                preference_data.append({
                    'prompt': prompt,
                    'prompt_idx': idx,
                    'winner_idx': real_winner_idx,
                    'loser_idx': real_loser_idx,
                    'metadata': metadata
                })
    
    return preference_data

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
    train_dataset = ClinicalDataset(
        args.clinical_train_metadata,
        args.clinical_train_images,
        resolution=args.resolution
    )
    val_dataset = ClinicalDataset(
        args.clinical_val_metadata,
        args.clinical_val_images,  
        resolution=args.resolution
    )
    
    # Load Stable Diffusion components
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
    
    # Setup precision
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    # Move models to device
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    
    # Initialize quality assessors
    assessors = QualityAssessors(
        args.disease_classifier_path,
        args.gender_classifier_path,
        args.race_classifier_path,
        args.age_classifier_path,
        accelerator.device
    )
    
    # Initialize consistency validator
    consistency_validator = ConsistencyValidator(accelerator.device)
    
    # Generate or load preference pairs
    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        logger.info("Preparing preference pairs...")
        pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=noise_scheduler,
            safety_checker=None,
            feature_extractor=None
        ).to(accelerator.device)
        
        preference_cache_path = os.path.join(args.output_dir, "preference_pairs.json")
        
        if os.path.exists(preference_cache_path):
            with open(preference_cache_path, 'r') as f:
                preference_data = json.load(f)
            logger.info(f"Loaded {len(preference_data)} cached preference pairs")
        else:
            logger.info("Generating new preference pairs...")
            preference_data = generate_preference_pairs(
                pipeline,
                train_dataset,
                assessors,
                args,
                accelerator.device
            )
            with open(preference_cache_path, 'w') as f:
                json.dump(preference_data, f, indent=2)
            logger.info(f"Generated and saved {len(preference_data)} preference pairs")
        
        del pipeline
        torch.cuda.empty_cache()
    
    accelerator.wait_for_everyone()
    
    # Load preference data on all processes
    preference_cache_path = os.path.join(args.output_dir, "preference_pairs.json")
    with open(preference_cache_path, 'r') as f:
        preference_data = json.load(f)
    
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
    unet, optimizer, lr_scheduler = accelerator.prepare(unet, optimizer, lr_scheduler)
    
    if accelerator.is_main_process:
        accelerator.init_trackers("skin-anofair-selfpo", config=vars(args))
    
    # Training info
    logger.info("***** Running LoRASelf-PO Training *****")
    logger.info(f"  Clinical dataset size = {len(train_dataset)}")
    logger.info(f"  Preference pairs = {len(preference_data)}")
    logger.info(f"  Max training steps = {args.max_train_steps}")
    logger.info(f"  DPO Beta = {args.beta}")
    logger.info(f"  LoRA Rank = {args.rank}")
    
    global_step = 0
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Training Steps")
    
    unet.train()
    
    # Training loop
    while global_step < args.max_train_steps:
        # Sample random preference pair
        pair = random.choice(preference_data)
        prompt_idx = pair['prompt_idx']
        
        data = train_dataset[prompt_idx]
        image = data['image'].unsqueeze(0).to(accelerator.device, dtype=weight_dtype)
        prompt = data['prompt']
        
        with accelerator.accumulate(unet):
            # Encode prompt
            text_inputs = tokenizer(
                [prompt],
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_embeddings = text_encoder(
                text_inputs.input_ids.to(accelerator.device)
            )[0]
            
            with torch.no_grad():
                latents_x = vae.encode(image).latent_dist.sample()
                latents_x = latents_x * vae.config.scaling_factor
            
            # Sample timestep
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, 
                (1,), device=accelerator.device
            )
            
            # === STEP 1: Prepare noise for winner and loser ===
            
            # Generate noise for winner with specific seed
            generator_w = torch.Generator(device=accelerator.device).manual_seed(
                args.seed + pair['winner_idx'] + prompt_idx * 1000
            )
            noise_w = torch.randn_like(latents_x, generator=generator_w)

            # Generate noise for loser with different seed
            generator_l = torch.Generator(device=accelerator.device).manual_seed(
                args.seed + pair['loser_idx'] + prompt_idx * 1000
            )
            noise_l = torch.randn_like(latents_x, generator=generator_l)

            # Concatenate latents and noise for batch processing
            latents_concat = torch.cat([latents_x, latents_x], dim=0)  # [2, C, H, W]
            noise_concat = torch.cat([noise_w, noise_l], dim=0)  # [2, C, H, W]
            text_embeddings_concat = text_embeddings.repeat(2, 1, 1)  # [2, seq_len, dim]

            # Sample timestep (same for both winner and loser)
            # Note: Using same timestep for fair comparison
            timesteps_single = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (1,), device=accelerator.device
            ).long()
            timesteps_concat = timesteps_single.repeat(2)  # [2]

            # Add noise to latents (forward diffusion)
            noisy_latents_concat = noise_scheduler.add_noise(
                latents_concat, noise_concat, timesteps_concat
            )

            # === STEP 2: Policy Model (LoRA enabled) Predictions ===
            
            model_pred_concat = unet(
                noisy_latents_concat,
                timesteps_concat,
                text_embeddings_concat
            ).sample

            # === STEP 3: Reference Model (LoRA disabled) Predictions ===
            
            # Temporarily disable LoRA adapters to get base model predictions
            unet_unwrapped = accelerator.unwrap_model(unet)
            unet_unwrapped.disable_adapters()
            
            with torch.no_grad():
                ref_pred_concat = unet(
                    noisy_latents_concat,
                    timesteps_concat,
                    text_embeddings_concat
                ).sample.detach()
            
            # Re-enable LoRA adapters
            unet_unwrapped.enable_adapters()

            # === STEP 4: Compute Target ===
            
            with torch.no_grad():
                if noise_scheduler.config.prediction_type == "epsilon":
                    target_concat = noise_concat
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target_concat = noise_scheduler.get_velocity(
                        latents_concat, noise_concat, timesteps_concat
                    )
                else:
                    raise ValueError(
                        f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                    )

            # === STEP 5: Compute MSE Losses ===
            
            # Policy model losses (LoRA enabled)
            model_losses = F.mse_loss(
                model_pred_concat.float(),
                target_concat.float(),
                reduction="none"
            )
            # Mean over spatial dimensions, keep batch dimension
            model_losses = model_losses.mean(dim=list(range(1, len(model_losses.shape))))  # [2]
            model_losses_w, model_losses_l = model_losses.chunk(2)  # [1], [1]
            
            # Reference model losses (base model, no LoRA)
            ref_losses = F.mse_loss(
                ref_pred_concat.float(),
                target_concat.float(),
                reduction="none"
            )
            # Mean over spatial dimensions, keep batch dimension
            ref_losses = ref_losses.mean(dim=list(range(1, len(ref_losses.shape))))  # [2]
            ref_losses_w, ref_losses_l = ref_losses.chunk(2)  # [1], [1]

            # === STEP 6: Compute DPO Logits (CORRECT!) ===
            
            # Reference model difference: winner vs loser
            ref_diff = ref_losses_w - ref_losses_l
            
            # Policy model difference: winner vs loser
            model_diff = model_losses_w - model_losses_l
            
            # DPO logits: reference advantage minus policy advantage
            # This measures how much better the policy is at discriminating
            # winner from loser compared to the reference model
            logits = ref_diff - model_diff

            # === STEP 7: Compute DPO Loss (Equation 7) ===
            
            # Standard DPO sigmoid loss
            loss = -F.logsigmoid(args.beta * logits).mean()

            # === STEP 8: Compute Implicit Accuracy (for logging) ===
            
            with torch.no_grad():
                # Accuracy: how often does the model prefer winner over loser?
                implicit_acc = (logits > 0).sum().float() / logits.size(0)
                implicit_acc += 0.5 * (logits == 0).sum().float() / logits.size(0)

            # === STEP 9: Backward Pass ===
            
            # Backward pass
            accelerator.backward(loss)
            
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(params_to_optimize, args.max_grad_norm)
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        # Logging and validation
        if accelerator.sync_gradients:
            progress_bar.update(1)
            global_step += 1
            
            # Log training metrics
            logs = {
                "train/loss": loss.detach().item(),
                "train/loss_winner": model_losses_w.detach().item(),
                "train/loss_loser": model_losses_l.detach().item(),
                "train/ref_loss_winner": ref_losses_w.detach().item(),  
                "train/ref_loss_loser": ref_losses_l.detach().item(),   
                "train/model_diff": model_diff.detach().item(),        
                "train/ref_diff": ref_diff.detach().item(),           
                "train/logits": logits.detach().item(),
                "train/implicit_accuracy": implicit_acc.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            # Validation
            if global_step % args.validation_steps == 0 and accelerator.is_main_process:
                logger.info(f"Running validation at step {global_step}...")
                
                # Create validation pipeline
                val_pipeline = StableDiffusionPipeline(
                    vae=vae,
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    unet=accelerator.unwrap_model(unet),
                    scheduler=noise_scheduler,
                    safety_checker=None,
                    feature_extractor=None
                ).to(accelerator.device)
                
                # Sample validation prompts
                val_prompts = [
                    val_dataset[i]['prompt'] 
                    for i in range(min(args.num_validation_prompts, len(val_dataset)))
                ]
                
                # Validate consistency
                mean_lpips, std_lpips = consistency_validator.validate_consistency(
                    val_pipeline, val_prompts, num_samples=3
                )
                
                logger.info(f"Step {global_step} - Validation LPIPS: {mean_lpips:.4f} ± {std_lpips:.4f}")
                logger.info(f"(Lower LPIPS = Better consistency)")
                
                accelerator.log({
                    "val/lpips_mean": mean_lpips,
                    "val/lpips_std": std_lpips
                }, step=global_step)
                
                del val_pipeline
                torch.cuda.empty_cache()
            
            # Checkpoint
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
            os.path.join(args.output_dir, "lora_self_po.safetensors"),
        )
        
        logger.info(f"LoRASelf-PO training completed!")
        logger.info(f"Model saved to {args.output_dir}/lora_self_po.safetensors")
    
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
