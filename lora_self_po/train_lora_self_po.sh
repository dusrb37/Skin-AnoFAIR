#!/bin/bash

# Skin-AnoFAIR: Train LoRASelf-PO
# Self-supervised preference optimization using clinical dataset only

export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATA_DIR="./data/skin-anofair"

OUTPUT_DIR="./models/lora_self_po"
mkdir -p $OUTPUT_DIR

# Dataset paths (clinical only)
CLINICAL_TRAIN="${DATA_DIR}/clinical/train_metadata.json"
CLINICAL_VAL="${DATA_DIR}/clinical/val_metadata.json"
CLINICAL_TRAIN_IMAGES="${DATA_DIR}/clinical/train"  
CLINICAL_VAL_IMAGES="${DATA_DIR}/clinical/val"     

# Pre-trained classifiers
DISEASE_CLASSIFIER="${DATA_DIR}/classifiers/disease_classifier.pt"
GENDER_CLASSIFIER="${DATA_DIR}/classifiers/gender_classifier.pt"
RACE_CLASSIFIER="${DATA_DIR}/classifiers/race_classifier_5class.pt"
AGE_CLASSIFIER="${DATA_DIR}/classifiers/age_classifier_2class.pt"

# Preference generation parameters
NUM_CANDIDATES=8      # k: candidates per prompt
NUM_HIGH_QUALITY=4    # m: high-quality subset

# Training parameters
TRAIN_BATCH_SIZE=1
LEARNING_RATE=1e-5
MAX_STEPS=3000
BETA=0.3             # beta parameter
RANK=32
ALPHA=16

# Generation parameters
INFERENCE_STEPS=50
GUIDANCE_SCALE=7.5

# Validation parameters
VALIDATION_STEPS=500
NUM_VAL_PROMPTS=10

# Install required packages
pip install lpips

# Run training
accelerate launch train_lora_self_po.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --output_dir=$OUTPUT_DIR \
  --clinical_train_metadata=$CLINICAL_TRAIN \
  --clinical_val_metadata=$CLINICAL_VAL \
  --clinical_train_images=$CLINICAL_TRAIN_IMAGES \
  --clinical_val_images=$CLINICAL_VAL_IMAGES \
  --disease_classifier_path=$DISEASE_CLASSIFIER \
  --gender_classifier_path=$GENDER_CLASSIFIER \
  --race_classifier_path=$RACE_CLASSIFIER \
  --age_classifier_path=$AGE_CLASSIFIER \
  --num_candidates=$NUM_CANDIDATES \
  --num_high_quality=$NUM_HIGH_QUALITY \
  --mixed_precision="fp16" \
  --train_batch_size=$TRAIN_BATCH_SIZE \
  --max_train_steps=$MAX_STEPS \
  --gradient_accumulation_steps=4 \
  --num_inference_steps=$INFERENCE_STEPS \
  --guidance_scale=$GUIDANCE_SCALE \
  --checkpointing_steps=500 \
  --validation_steps=$VALIDATION_STEPS \
  --num_validation_prompts=$NUM_VAL_PROMPTS \
  --learning_rate=$LEARNING_RATE \
  --lr_scheduler="constant" \
  --lr_warmup_steps=100 \
  --beta=$BETA \
  --adam_beta1=0.9 \
  --adam_beta2=0.999 \
  --adam_weight_decay=0.01 \
  --max_grad_norm=1.0 \
  --rank=$RANK \
  --lora_alpha=$ALPHA \
  --resolution=512 \
  --seed=42 \
  --gradient_checkpointing \
  --report_to="tensorboard" \
  --logging_dir="$OUTPUT_DIR/logs"