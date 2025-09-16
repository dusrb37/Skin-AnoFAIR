#!/bin/bash
# Skin-AnoFAIR: Train LoRAbias
# Goal: Learn inherent statistical bias of the dataset

export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATA_DIR="./data/skin-anofair"

OUTPUT_DIR="./models/lora_bias"
mkdir -p $OUTPUT_DIR

# Dataset paths
CLINICAL_TRAIN="${DATA_DIR}/clinical/train_metadata.json"
CLINICAL_VAL="${DATA_DIR}/clinical/val_metadata.json"

# Training parameters
TRAIN_BATCH_SIZE=8
GRADIENT_ACCUMULATION=1
LEARNING_RATE=5e-5  
MAX_STEPS=5000
RANK=32  
ALPHA=16

# Generation parameters
INFERENCE_STEPS=50
GUIDANCE_SCALE=7.5

# Logging
VALIDATION_STEPS=200
CHECKPOINT_STEPS=500

echo "========================================="
echo "Skin-AnoFAIR: LoRAbias Training"
echo "========================================="
echo "Base Model: $MODEL_NAME"
echo "Output Directory: $OUTPUT_DIR"
echo "----------------------------------------"
echo "Dataset:"
echo "  Clinical Train: $CLINICAL_TRAIN"
echo "  Clinical Val: $CLINICAL_VAL"
echo "----------------------------------------"
echo "Training Configuration:"
echo "  Batch size: $TRAIN_BATCH_SIZE"
echo "  Gradient accumulation: $GRADIENT_ACCUMULATION"
echo "  Learning rate: $LEARNING_RATE"
echo "  Max steps: $MAX_STEPS"
echo "  LoRA Rank: $RANK"
echo "  LoRA Alpha: $ALPHA"
echo "----------------------------------------"
echo "Generation:"
echo "  Inference steps: $INFERENCE_STEPS"
echo "  Guidance scale: $GUIDANCE_SCALE"
echo "----------------------------------------"
echo "Logging:"
echo "  Validation every: $VALIDATION_STEPS steps"
echo "  Checkpointing every: $CHECKPOINT_STEPS steps"
echo "========================================="

# Launch training
accelerate launch train_lora_bias.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --output_dir=$OUTPUT_DIR \
  --clinical_train_metadata=$CLINICAL_TRAIN \
  --clinical_val_metadata=$CLINICAL_VAL \
  --mixed_precision="fp16" \
  --train_batch_size=$TRAIN_BATCH_SIZE \
  --gradient_accumulation_steps=$GRADIENT_ACCUMULATION \
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
echo "Model saved to: $OUTPUT_DIR/lora_bias.safetensors"
echo "========================================="