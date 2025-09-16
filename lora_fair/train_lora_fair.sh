#!/bin/bash

# Skin-AnoFAIR: Train LoRAfair

export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATA_DIR="./data/skin-anofair"

OUTPUT_DIR="./models/lora_fair"
mkdir -p $OUTPUT_DIR

# Dataset paths
CLINICAL_TRAIN="${DATA_DIR}/clinical/train_metadata.json"
CLINICAL_VAL="${DATA_DIR}/clinical/val_metadata.json"
FAIRFACE_TRAIN="${DATA_DIR}/fairface/train_metadata.json"
FAIRFACE_VAL="${DATA_DIR}/fairface/val_metadata.json"
CLINICAL_TRAIN_IMAGES="${DATA_DIR}/clinical/train"
CLINICAL_VAL_IMAGES="${DATA_DIR}/clinical/val"
FAIRFACE_TRAIN_IMAGES="${DATA_DIR}/fairface/train"
FAIRFACE_VAL_IMAGES="${DATA_DIR}/fairface/val"  # 추가

# Pre-trained classifier paths
GENDER_CLASSIFIER="${DATA_DIR}/classifiers/gender_classifier.pt"
RACE_CLASSIFIER="${DATA_DIR}/classifiers/race_classifier_5class.pt"
AGE_CLASSIFIER="${DATA_DIR}/classifiers/age_classifier_2class.pt"

# Training hyperparameters
TRAIN_BATCH_SIZE=1
IMAGES_PER_PROMPT=16  # For distributional alignment
LEARNING_RATE=5e-5
MAX_STEPS=5000
RANK=32
ALPHA=16

# Generation parameters
INFERENCE_STEPS=20
GUIDANCE_SCALE=7.5

# Logging
VALIDATION_STEPS=500
CHECKPOINT_STEPS=500

accelerate launch train_lora_fair.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --output_dir=$OUTPUT_DIR \
  --clinical_train_metadata=$CLINICAL_TRAIN \
  --clinical_val_metadata=$CLINICAL_VAL \
  --fairface_train_metadata=$FAIRFACE_TRAIN \
  --fairface_val_metadata=$FAIRFACE_VAL \
  --clinical_train_images=$CLINICAL_TRAIN_IMAGES \
  --clinical_val_images=$CLINICAL_VAL_IMAGES \
  --fairface_train_images=$FAIRFACE_TRAIN_IMAGES \
  --fairface_val_images=$FAIRFACE_VAL_IMAGES \
  --gender_classifier_path=$GENDER_CLASSIFIER \
  --race_classifier_path=$RACE_CLASSIFIER \
  --age_classifier_path=$AGE_CLASSIFIER \
  --mixed_precision="fp16" \
  --train_batch_size=$TRAIN_BATCH_SIZE \
  --images_per_prompt=$IMAGES_PER_PROMPT \
  --max_train_steps=$MAX_STEPS \
  --num_inference_steps=$INFERENCE_STEPS \
  --guidance_scale=$GUIDANCE_SCALE \
  --checkpointing_steps=$CHECKPOINT_STEPS \
  --validation_steps=$VALIDATION_STEPS \
  --learning_rate=$LEARNING_RATE \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
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