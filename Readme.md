# EuroSAT Land Cover Classifier — ResNet50 Fine-Tuning

A transfer learning project that fine-tunes a pretrained **ResNet50** to classify Sentinel-2 satellite image patches into 10 land cover categories, using the **EuroSAT** dataset. Includes a bonus real-world application: using the trained model to detect land-use change over time in a specific region.

🤗 **Trained model on Hugging Face**: [chitransh001/euroset-resnet50](https://huggingface.co/chitransh001/euroset-resnet50)

---

## Overview

This project takes a ResNet50 pretrained on ImageNet and adapts it to satellite imagery through a two-stage transfer learning approach:

1. **Frozen backbone training** — freeze all of ResNet50's pretrained layers, train only a new final classification layer.
2. **Full fine-tuning** — unfreeze the entire network and continue training at a much lower learning rate, letting every layer slightly adjust to satellite imagery.

## Dataset

- **[EuroSAT](https://github.com/phelber/EuroSAT)** — 27,000+ Sentinel-2 satellite image patches (RGB, 64×64 px, 10m/pixel resolution)
- **10 land cover classes**: AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake
- Used the official Kaggle packaging ([`apollo2506/eurosat-dataset`](https://www.kaggle.com/datasets/apollo2506/eurosat-dataset)) with a predefined, reproducible **70/20/10 train/val/test split** (18,900 / 5,400 / 2,700 images)

## Model & Approach

| Stage | What's trained | Learning rate | Result (val accuracy) |
|---|---|---|---|
| 1. Frozen backbone | Final FC layer only | 1e-3 | 94% |
| 2. Full fine-tune | Entire network | 1e-4 | ~98.1–98.3% |

- **Base model**: `torchvision.models.resnet50`, `IMAGENET1K_V2` pretrained weights
- **Preprocessing**: resize to 224×224, normalize with ImageNet mean/std
- **Augmentation** (train only): random horizontal + vertical flip — satellite imagery has no inherent "up," so flips are a natural fit
- **Framework**: PyTorch, trained on Google Colab (T4 GPU)

## Results (Test Set, 2,700 images)

**Overall accuracy: 98%**

| Class | F1-score |
|---|---|
| Residential | 1.00 |
| SeaLake | 1.00 |
| Forest | 0.99 |
| Highway | 0.99 |
| Industrial | 0.99 |
| HerbaceousVegetation | 0.98 |
| Pasture | 0.97 |
| River | 0.97 |
| AnnualCrop | 0.96 |
| PermanentCrop | 0.96 |

Main confusion patterns observed: agricultural sub-classes (AnnualCrop / PermanentCrop / Pasture) overlapping with each other, and an unexpected River → AnnualCrop misclassification, likely from narrow/vegetation-lined river sections resembling cropland at 64×64 resolution.

## Bonus: Applied Land-Use Change Detection

The trained classifier was applied to real Sentinel-2 imagery of **Noida, India** across two time periods (2017 vs. 2024) to detect land-use change:

- Pulled and preprocessed cloud-free Sentinel-2 composites via Google Earth Engine
- Patched each composite into 64×64 tiles matching the model's input format
- Classified every patch for both years and built a transition matrix comparing 2017 → 2024
- **Finding**: significant conversion of agricultural and forested land into industrial/urban use — ~28% of all patches shifted from non-urban to urban infrastructure (Industrial/Residential/Highway) over the 7-year span
- Cross-checked Forest-loss patches against Google Earth Engine's Hansen Global Forest Change dataset — found a large discrepancy explained by a definitional mismatch (Hansen's dense-canopy threshold doesn't register the sparser, fragmented vegetation common in semi-urban Indian landscapes)

## How to Use the Trained Model

```python
import torch
import torchvision.models as models
import torch.nn as nn
from huggingface_hub import hf_hub_download

weights_path = hf_hub_download(repo_id="chitransh001/euroset-resnet50", filename="eurosat_resnet50.pt")

model = models.resnet50(weights='IMAGENET1K_V2')
model.fc = nn.Linear(model.fc.in_features, 10)
model.load_state_dict(torch.load(weights_path, map_location='cpu'))
model.eval()
```

Label map:
```python
{'AnnualCrop': 0, 'Forest': 1, 'HerbaceousVegetation': 2, 'Highway': 3, 'Industrial': 4,
 'Pasture': 5, 'PermanentCrop': 6, 'Residential': 7, 'River': 8, 'SeaLake': 9}
```

## Tech Stack

- PyTorch / torchvision
- Google Colab (T4 GPU)
- Google Earth Engine (satellite imagery + change detection)
- scikit-learn (evaluation metrics)
- Hugging Face Hub (model hosting)

## Repository Contents

- `eurosat_classifier.ipynb` — full training, evaluation, and land-use change detection pipeline
- `confusion_matrix.png` — test set confusion matrix

## Credits

- EuroSAT dataset: Helber et al., 2019, *IEEE JSTARS*
- Pretrained weights: torchvision (`IMAGENET1K_V2`)
- Forest cover validation: Hansen et al., *Global Forest Change* (via Google Earth Engine)
