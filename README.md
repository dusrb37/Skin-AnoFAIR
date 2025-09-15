# Skin-AnoFAIR: Anonymization of Facial Skin Disease Images with Bias Mitigation using Specialist Adapters

Official implementation of "Skin-AnoFAIR: Anonymization of Facial Skin Disease Images with Bias Mitigation using Specialist Adapters"

<div align="center">
  <img src="./assets/Figure1.png" width="100%">
</div>

## Overview

**Skin-AnoFAIR** presents a novel approach for anonymizing facial skin disease images while mitigating demographic biases through specialist adapters (LoRA modules). Our three-stage pipeline ensures:

1. **LoRAbias**: Preserves disease-specific features with demographic conditioning
2. **LoRAfair**: Achieves demographic fairness through distributional alignment
3. **LoRASelf-PO**: Enhances generation quality via self-supervised preference optimization

## Three-Stage Pipeline

| Stage | Objective | Training Data | Key Method |
|-------|-----------|---------------|------------|
| [LoRAbias](./lora_bias/) | Disease feature preservation | Clinical (6k) | Cross-attention control |
| [LoRAfair](./lora_fair/) | Demographic fairness | Clinical + FairFace (16k) | Distributional alignment |
| [LoRASelf-PO](./lora_self_po/) | Quality refinement | Clinical (6k) | Self-supervised DPO |



## Installation
```bash
conda create -n Skin-AnoFAIR python=3.10.13
conda activate Skin-AnoFAIR

git clone https://github.com/dusrb37/skin-anofair.git
cd skin-anofair
pip install peft
pip install diffusers
pip install -r requirements.txt

