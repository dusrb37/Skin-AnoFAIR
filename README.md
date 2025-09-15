# Skin-AnoFAIR: Anonymization of Facial Skin Disease Images with Bias Mitigation using Specialist Adapters

Official implementation of "Skin-AnoFAIR: Anonymization of Facial Skin Disease Images with Bias Mitigation using Specialist Adapters"

<div align="center">
  <img src="./assets/Figure1.png" width="100%">
</div>

## Overview

**Skin-AnoFAIR** presents a novel approach for anonymizing facial skin disease images while mitigating demographic biases through specialist adapters (LoRA modules):

1. **LoRA_bias**: Preserves disease-specific features with demographic conditioning
2. **LoRA_fair**: Achieves demographic fairness through distributional alignment
3. **LoRA_Self-PO**: Enhances generation quality via self-supervised preference optimization


| Stage | Objective |
|-------|-----------|
| [LoRA_bias](./lora_bias/) | Disease feature preservation |
| [LoRA_fair](./lora_fair/) | Demographic fairness |
| [LoRA_Self-PO](./lora_self_po/) | Quality refinement |



## Installation
```bash
conda create -n Skin-AnoFAIR python=3.10.13
conda activate Skin-AnoFAIR

git clone https://github.com/dusrb37/skin-anofair.git
cd skin-anofair
pip install peft
pip install diffusers
pip install -r requirements.txt

## Quick Start
Train the three specialist adapters sequentially:

```bash
# Stage 1: LoRAbias - Disease feature preservation
cd lora_bias
bash train_lora_bias.sh

# Stage 2: LoRAfair - Demographic fairness
cd ../lora_fair
bash train_lora_fair.sh

# Stage 3: LoRASelf-PO - Quality enhancement
cd ../lora_self_po
bash train_lora_self_po.sh

## Data Preparation
Due to IRB restrictions and patient privacy, we cannot share the clinical facial skin disease dataset. Required data:

Clinical dataset: 6,000 facial skin disease images with annotations
FairFace dataset: Public dataset for demographic balance (Stage 2 only)
Pre-trained classifiers: Disease, demographic (gender, race, age)

See individual stage READMEs for detailed specifications.





