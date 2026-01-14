# Plant Disease Classification using EfficientNet + SVM

This project classifies plant leaf diseases using **deep features extracted from EfficientNet-B0** combined with an **SVM classifier**. It is trained and evaluated on the [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease/data).

---

## **Features**

- Loads and preprocesses images from the PlantVillage dataset.
- Uses **EfficientNet-B0** as a feature extractor (pretrained on ImageNet).
- Extracts deep features from images and caches them for faster SVM training.
- Trains an **SVM classifier** with hyperparameter tuning using **HalvingGridSearchCV**.
- Visualizes:
  - Class distribution
  - Sample images
  - Confusion matrix of predictions
- Saves trained models:
  - `feature_extractor.pth` for EfficientNet features
  - `svm_plant_disease_model.pkl` for the SVM classifier
- Handles missing/corrupt images gracefully during loading.

---

## **Project Structure**
plant-disease-classification/
│
├─ data/                        # Dataset and cached models
│   ├─ features_cache.pkl        # Cached features for SVM
│   ├─ label_encoder.pkl         # Label encoder for classes
│   ├─ svm_plant_disease_model.pkl # Trained SVM model
│   └─ feature_extractor.pth     # EfficientNet feature extractor
│
├─ plant_leaf_disease_detection.py                   # Main training & evaluation script
├─ requirements.txt              # Python dependencies
└─ README.md

