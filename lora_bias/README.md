# LoRA_bias - Baseline Statistical Bias Learning

First adapter that learns the inherent statistical distribution of the clinical dataset without any conditioning.

## Objective

LoRA_bias captures the natural statistical biases present in the clinical skin disease dataset as a baseline model.

## Method

- **Unconditional learning**: No demographic or disease conditioning
- **Statistical bias capture**: Learns inherent dataset distribution
- **Baseline reference**: Serves as comparison for fairness evaluation

## Data Structure
```bash
data/
└── clinical/
    ├── train/          # Training images
    │   └── ...
    └── val/            # Validation images
        └── ...
```

## Training
```bash
bash train_lora_bias.sh
