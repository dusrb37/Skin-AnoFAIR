# LoRA_Self-PO - Self-Supervised Preference Optimization

Final specialist adapter enhancing generation quality through self-supervised preference learning.

## Objective

LoRA_Self-PO refines output quality using self-discovered preferences without human annotations.

## Method

Self-supervised preference discovery:
1. Generate k=8 candidates per prompt
2. Quality assessment via 5 models
3. Select top-m=4 high-quality subset
4. Compute DINOv2 feature centroid
5. Train 

## Data Structure
```bash
data/
├── clinical/
│   ├── train/               
│   ├── val/                 
│   ├── train_metadata.json  
│   └── val_metadata.json    
└── classifiers/
    ├── disease_classifier.pt
    ├── gender_classifier.pt
    ├── race_classifier_5class.pt
    └── age_classifier_2class.pt
```

## Training
```bash
bash train_lora_self_po.sh