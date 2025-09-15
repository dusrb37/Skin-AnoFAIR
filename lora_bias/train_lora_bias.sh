#!/bin/bash

# Skin-AnoFAIR Stage 1: Train LoRAbias
# Goal: Learn inherent statistical bias of the dataset

# Model path
export MODEL_NAME="runwayml/stable-diffusion-v1-5"

# Dataset path
export TRAIN_DATA_DIR="/path/to/facial_skin_disease_dataset/train"

# Output directory
OUTPUT_DIR="./models/lora_bias"
mkdir -p $OUTPUT_DIR

# Training parameters
BATCH_SIZE=8
GRADIENT_ACCUMULATION=1
LEARNING_RATE=5e-5  
EPOCHS=10
RANK=32  
ALPHA=16

# Validation
VALIDATION_STEPS=200

echo "========================================="
echo "Skin-AnoFAIR Stage 1: LoRAbias Training"
echo "========================================="
echo "Model: $MODEL_NAME"
echo "Dataset: $TRAIN_DATA_DIR"
echo "Output: $OUTPUT_DIR"
echo "LoRA Rank: $RANK, Alpha: $ALPHA"
echo "Learning Rate: $LEARNING_RATE"
echo "Validation every: $VALIDATION_STEPS steps"
echo "========================================="

# Launch training
accelerate launch train_lora_bias.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DATA_DIR \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="fp16" \
  --train_batch_size=$BATCH_SIZE \
  --gradient_accumulation_steps=$GRADIENT_ACCUMULATION \
  --num_train_epochs=$EPOCHS \
  --checkpointing_steps=500 \
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

echo "========================================="
echo "LoRAbias training completed!"
echo "Saved to: $OUTPUT_DIR/lora_bias.safetensors"
echo "========================================="