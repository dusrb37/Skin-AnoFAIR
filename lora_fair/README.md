# LoRA_fair - Distributional Alignment for Demographic Fairness

Second specialist adapter achieving demographic fairness through distributional alignment.

## Objective

LoRA_fair mitigates demographic biases by aligning generated distributions with balanced target distributions.

## Method

- **Distributional alignment**: Batch-level demographic distribution matching
- **Combined datasets**: Clinical + FairFace for comprehensive coverage
- **Multi-attribute fairness**: Simultaneous alignment of gender, race, and age
- **Efficient training**: Gradient hooks for memory optimization

## Data Structure
```bash
data/
├── clinical/
│   ├── train/              
│   ├── val/                 
│   └── train_metadata.json
├── fairface/
│   ├── train/ 
│   └── train_metadata.json # 10,000 demographically balanced
└── classifiers/
    ├── gender_classifier.pt       # 2 classes
    ├── race_classifier_5class.pt  # 5 classes
    └── age_classifier_2class.pt   # 2 classes
```

## Training
```bash
bash train_lora_fair.sh
