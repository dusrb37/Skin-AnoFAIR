# LoRA_bias - Disease Feature Preservation with Demographic Conditioning

First specialist adapter focusing on preserving disease-specific features while maintaining demographic representation.

## Objective

LoRA_bias ensures accurate disease feature preservation through cross-attention control while conditioning on demographic attributes.

## Method

- **Cross-attention control**: Guides disease-specific feature generation
- **Demographic conditioning**: Maintains representation across gender and race
- **Disease validation**: Pre-trained classifier ensures feature fidelity

## Data Structure
```bash
data/
├── clinical/
│   ├── train_metadata.json 
│   ├── val_metadata.json
│   └── test_metadata.json
└── classifiers/
    └── disease_classifier.pt  # 6 disease classes
```

## Metadata Format
```bash
{
  "image_id": "clinical_00001",
  "age": 30,
  "gender": "male",
  "race": "asian",
  "disease": "psoriasis"
}
```

## Training
```bash
bash train_lora_bias.sh
