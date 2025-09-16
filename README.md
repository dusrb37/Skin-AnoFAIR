# Skin-AnoFAIR: Anonymization of Facial Skin Disease Images with Bias Mitigation using Specialist Adapters

Official implementation of "Skin-AnoFAIR: Anonymization of Facial Skin Disease Images with Bias Mitigation using Specialist Adapters"

<div align="center">
  <img src="./assets/Figure1.png" width="100%">
</div>

## Overview

**Skin-AnoFAIR** presents a novel approach for anonymizing facial skin disease images while mitigating demographic biases through specialist adapters (LoRA modules):

1. **LoRA_bias**: captures the natural statistical biases present in the clinical skin disease dataset
2. **LoRA_fair**: Achieves demographic fairness through distributional alignment
3. **LoRA_Self-PO**: Enhances generation quality and generation consistency via self-supervised preference optimization


| Stage | Objective |
|-------|-----------|
| [LoRA_bias](./lora_bias/) | Baseline Statistical Bias Learning |
| [LoRA_fair](./lora_fair/) | Demographic fairness |
| [LoRA_Self-PO](./lora_self_po/) | Quality refinement-Generation consistency |



## Installation
```bash
conda create -n Skin-AnoFAIR python=3.10.13
conda activate Skin-AnoFAIR

git clone https://github.com/dusrb37/skin-anofair.git
cd skin-anofair
pip install peft
pip install diffusers
pip install -r requirements.txt
```

## Quick Start
### Train the specialist adapters sequentially:

```bash
# LoRA_bias - Disease feature preservation
cd lora_bias
bash train_lora_bias.sh

# LoRA_fair - Demographic fairness
cd lora_fair
bash train_lora_fair.sh

# LoRA_Self-PO - Quality enhancement, Generation consistency
cd lora_self_po
bash train_lora_self_po.sh
```
See individual stage READMEs for detailed specifications.

<be>

## Inference

To inference, Checkout - `inference.py` for mode details.






