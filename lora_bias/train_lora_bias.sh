#!/bin/bash
# Skin-AnoFAIR: Train LoRAbias
# Goal: Learn inherent statistical bias of the dataset

export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATA_DIR="./data/skin-anofair"

OUTPUT_DIR="./models/lora_bias"
mkdir -p $OUTPUT_DIR

# Dataset paths
TRAIN_DATA_DIR="${DATA_DIR}/clinical/train"
VAL_DATA_DIR="${DATA_DIR}/clinical/val"

# Training parameters
TRAIN_BATCH_SIZE=8
GRADIENT_ACCUMULATION=1
LEARNING_RATE=5e-5  
MAX_STEPS=5000
RANK=32  
ALPHA=16

# Logging
VALIDATION_STEPS=200
CHECKPOINT_STEPS=500

# Launch training
accelerate launch train_lora_bias.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --output_dir=$OUTPUT_DIR \
  --train_data_dir=$TRAIN_DATA_DIR \
  --validation_data_dir=$VAL_DATA_DIR \
  --mixed_precision="fp16" \
  --train_batch_size=$TRAIN_BATCH_SIZE \
  --gradient_accumulation_steps=$GRADIENT_ACCUMULATION \
  --max_train_steps=$MAX_STEPS \
  --checkpointing_steps=$CHECKPOINT_STEPS \
  --validation_steps=$VALIDATION_STEPS \
  --learning_rate=$LEARNING_RATE \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --adam_beta1=0.9 \
  --adam_beta2=0.999 \
  --adam_weight_decay=0.01 \
  --adam_epsilon=1e-08 \
  --max_grad_norm=1.0 \
  --rank=$RANK \
  --lora_alpha=$ALPHA \
  --resolution=512 \
  --seed=42 \
  --gradient_checkpointing \
  --use_8bit_adam \
  --dataloader_num_workers=4 \
  --report_to="tensorboard" \
  --logging_dir="$OUTPUT_DIR/logs"
