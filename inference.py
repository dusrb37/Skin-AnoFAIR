import PIL
import peft
import copy
import torch
import random
import os, sys
import argparse
import requests
from io import BytesIO
from IPython.display import display
from torchvision.ops import box_convert
from PIL import Image, ImageDraw, ImageFont
from huggingface_hub import hf_hub_download
from diffusers import StableDiffusionInpaintPipeline, UNet2DConditionModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sd_pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting", 
    torch_dtype=torch.float16,
    safety_checker=None,
).to(device)

LORA_PATHS = {
    'bias': "./lora_bias.safetensors",
    'fair': "./lora_fair.safetensors",
    'self_po': "./lora_self_po.safetensors"
}

try:
    sd_pipe.load_lora_weights(LORA_PATHS['bias'], adapter_name="lora_bias")
    sd_pipe.load_lora_weights(LORA_PATHS['fair'], adapter_name="lora_fair")
    sd_pipe.load_lora_weights(LORA_PATHS['self_po'], adapter_name="lora_self_po")
    print("All LoRA adapters loaded successfully")
    USE_LORA = True
except:
    print("Warning: Could not load LoRA adapters. Using base model only.")
    USE_LORA = False

def set_adapter_weights(pipe, alpha=0.5, beta=0.5, use_lora=True):

    if use_lora:
        adapters = ["lora_bias", "lora_fair", "lora_self_po"]
        
        # Correct weight calculation
        w_bias = 1.0 - alpha - beta
        weights = [w_bias, alpha, beta]
        
        try:
            pipe.set_adapters(adapters, adapter_weights=weights)
            print(f"Adapter weights set - Bias: {w_bias:.2f}, Fair: {alpha:.2f}, Self-PO: {beta:.2f}")
        except:
            print("Using base model without adapters")
    else:
        print("LoRA not available, using base model")
    
    return pipe

def generate_skin_disease_prompt(disease_name, target_attributes=None):
    if target_attributes is None:
        target_attributes = {
            'race': 'asian',
            'gender': 'person',
            'age': '30'
        }
    
    race = target_attributes.get('race', 'asian')
    gender = target_attributes.get('gender', 'person')
    age = target_attributes.get('age', '30')
    
    prompt = f"A {age} years old {race} {gender} with {disease_name}."
    
    return prompt

def generate_anonymized_image(image_path, mask_path, disease_name, 
                              target_attributes=None,
                              alpha=0.5, beta=0.5,
                              pipe=None, seed=None):
    try:
        in_image = Image.open(image_path).convert("RGB")
        in_mask = Image.open(mask_path).convert("L")
        
        in_image = in_image.resize((512, 512), Image.LANCZOS)
        in_mask = in_mask.resize((512, 512), Image.LANCZOS)
    except IOError as e:
        print(f"Loading error: {e}")
        return None
    
    prompt = generate_skin_disease_prompt(disease_name, target_attributes)
    print(f"Generated prompt: {prompt}")
    
    negative_prompt = "low quality, distorted, disfigured, unnatural skin texture, blurry, artifacts"
    
    if seed is None:
        seed = random.getrandbits(32)
    generator = torch.Generator(device).manual_seed(seed)
    
    pipe = set_adapter_weights(pipe, alpha=alpha, beta=beta, use_lora=USE_LORA)
    
    result = pipe(
        image=in_image,
        mask_image=in_mask,
        prompt=prompt,
        negative_prompt=negative_prompt,
        generator=generator,
        num_inference_steps=50,
        guidance_scale=7.5
    )
    
    return result.images[0]

if __name__ == "__main__":

    image_path = '/data/clinical_00001.png'
    mask_path = "/data/clinical_00001_mask.png"
    
    # Settings
    generated_image = generate_anonymized_image(
        image_path=image_path,
        mask_path=mask_path,
        disease_name="psoriasis",
        target_attributes={
            'race': 'asian',
            'gender': 'female',
            'age': '30'
        },
        alpha=0.5,  # fairness control
        beta=0.5,   # consistency control
        pipe=sd_pipe
    )
    
    if generated_image:
        generated_image.save("result_default.png")
        display(generated_image)
    

    # Testing different alpha values
    for alpha_val in [0.0, 0.3, 0.7, 1.0]:
        print(f"Generating with α={alpha_val}")
        
        result = generate_anonymized_image(
            image_path=image_path,
            mask_path=mask_path,
            disease_name="acne",
            target_attributes={
                'race': 'white',
                'gender': 'male',
                'age': '20'
            },
            alpha=alpha_val,  
            beta=0.5,         
            pipe=sd_pipe
        )
        
        if result:
            result.save(f"result_alpha_{alpha_val}.png")
            print(f"Saved: result_alpha_{alpha_val}.png")
    

    # Different demographic targets
    demographic_targets = [
        {'race': 'asian', 'gender': 'female', 'age': '10'},
        {'race': 'asian', 'gender': 'male', 'age': '30'},
        {'race': 'black', 'gender': 'female', 'age': '60'},
        {'race': 'white', 'gender': 'male', 'age': '50'}
    ]
    
    for i, target_attr in enumerate(demographic_targets):
        print(f"Target {i+1}: {target_attr}")
        
        result = generate_anonymized_image(
            image_path=image_path,
            mask_path=mask_path,
            disease_name="rosacea",
            target_attributes=target_attr,
            alpha=0.7,
            beta=0.5,
            pipe=sd_pipe
        )
        
        if result:
            filename = f"result_{target_attr['race']}_{target_attr['gender']}_{target_attr['age']}.png"
            result.save(filename)
            print(f"Saved: {filename}")
    

    # Testing different beta values
    for beta_val in [0.0, 0.3, 0.7, 1.0]:
        print(f"Generating with β={beta_val}")
        
        result = generate_anonymized_image(
            image_path=image_path,
            mask_path=mask_path,
            disease_name="seborrheic_dermatitis",
            target_attributes={
                'race': 'asian',
                'gender': 'female',
                'age': '40'
            },
            alpha=0.5,        
            beta=beta_val,    
            pipe=sd_pipe
        )
        
        if result:
            result.save(f"result_beta_{beta_val}.png")
            print(f"Saved: result_beta_{beta_val}.png")