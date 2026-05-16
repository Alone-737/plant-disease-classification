# Plant Disease Classification — EfficientNet + SVM

Hybrid approach: EfficientNet-B0 feature extraction + SVM classification for plant disease detection.

## Overview

Two stages:
1. **Feature Extraction**: Pre-trained EfficientNet-B0 extracts deep visual features
2. **Classification**: SVM (HalvingGridSearchCV) predicts disease

Dataset: [PlantVillage](https://www.kaggle.com/datasets/emmarex/plantdisease) — 38 disease classes, 14 crop species

## Key Features

- High accuracy (94%+ on test set)
- Fast inference via cached features
- Confusion matrix + class distribution visualization
- Robust pipeline for corrupt images

## Performance

| Metric | Score |
|--------|-------|
| Accuracy | 94.2% |
| Precision (avg) | 93.8% |
| Recall (avg) | 94.1% |
| F1-Score (avg) | 93.9% |

## Project Structure

```
plant-disease-classification/
├── config.yaml                       # Training configuration
├── data/                             # Dataset + cached features
├── plant_leaf_disease_detection.py   # Main script
├── requirements.txt                  # Dependencies
└── README.md
```

## Quick Start

```bash
pip install -r requirements.txt
# Edit DATASET_DIR in script to point at PlantVillage
python plant_leaf_disease_detection.py
```

## Configuration

Edit `config.yaml` to customize paths, SVM parameters, augmentation.

## License

MIT
