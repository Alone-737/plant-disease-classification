import cv2
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
import pickle
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import  confusion_matrix ,accuracy_score
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

import torch
from torch import nn
from torch.cuda.amp import autocast
import torchvision
from torchvision.transforms import transforms
from torch.utils.data import DataLoader,Dataset






DATASET_DIR = os.path.join("plant disease classification", "data", "PlantVillage dataset")
class_names = os.listdir(DATASET_DIR)
print(len(class_names), class_names)





class_counts={cls:len(os.listdir(os.path.join(DATASET_DIR,cls)))for cls in class_names}
plt.figure(figsize=(10, 4))
sns.barplot(x=list(class_counts.keys()),y=list(class_counts.values()))
plt.xticks(rotation=90)
plt.title("Class Distribution in Dataset")
plt.show()





def show_sample_images():
    plt.figure(figsize=(12,6))
    image_exts = ('.jpg', '.jpeg', '.png')
    for i,cls in enumerate(np.random.choice(class_names,6)):
        class_dir=os.path.join(DATASET_DIR,cls)
        images = [f for f in os.listdir(class_dir) if f.lower().endswith(image_exts)]
        if len(images) == 0:
            continue
        img_name=np.random.choice(images)
        img_path=os.path.join(class_dir,img_name)
        img=cv2.imread(img_path)
        if img is None:
            continue
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        plt.subplot(2,3,i+1)
        plt.imshow(img)
        plt.title(cls)  
        plt.axis('off')
    plt.show()






class load_image(Dataset):
    def __init__(self,image_paths,labels,transform=None):
        self.image_paths=image_paths
        self.labels=labels
        self.transform=transform
    def __getitem__(self, idx):
        max_retries=3
        for attempt in range(max_retries):
            try:
                img = cv2.imread(self.image_paths[idx])
                if img is None:
                    raise ValueError(f"Failed to load: {self.image_paths[idx]}")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if self.transform:
                    img = self.transform(img)     
                else:
                    img = cv2.resize(img, (224, 224))   
                    img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                return img, self.labels[idx]
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Error loading {self.image_paths[idx]}: {e}")
                    return torch.zeros(3, 224, 224), self.labels[idx]

    def __len__(self):
        return len(self.image_paths)
image_paths = []
labels = []
for class_name in class_names:
    class_dir = os.path.join(DATASET_DIR, class_name)
    for img_name in os.listdir(class_dir):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_paths.append(os.path.join(class_dir, img_name))
            labels.append(class_name)
le=LabelEncoder()
encoded_labels=le.fit_transform(labels)
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print(f"Total images: {len(image_paths)}")
print(f"Number of classes: {len(class_names)}")





transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

X_train, X_test, y_train, y_test = train_test_split(
    image_paths, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
)


train_dataset = load_image(X_train, y_train, transform=transform_train)
test_dataset = load_image(X_test, y_test, transform=transform_test)
num_workers=0
train_loader = DataLoader(train_dataset, batch_size=64, num_workers=num_workers, shuffle=True,
    pin_memory=True if torch.cuda.is_available() else False,
    persistent_workers=False
)
test_loader = DataLoader(test_dataset, batch_size=64, num_workers=num_workers, shuffle=False,
    pin_memory=True if torch.cuda.is_available() else False,
    persistent_workers=False
)
print(f"Training samples: {len(train_dataset)}")
print(f"Testing samples: {len(test_dataset)}")




base_model=torchvision.models.efficientnet_b0(weights="IMAGENET1K_V1")
base_model=base_model.features





class EfficientNetFeatureExtractor(nn.Module):
    def __init__(self,base_model):
        super().__init__()
        self.base_model=base_model
        self.pool=nn.AdaptiveAvgPool2d((1,1))

    def forward(self,x):
        x=self.base_model(x)
        x=self.pool(x)
        x=torch.flatten(x,1)
        return x

feature_extractor=EfficientNetFeatureExtractor(base_model)
feature_extractor.eval()
for param in feature_extractor.parameters():
    param.requires_grad=False





preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
CACHE_FILE = "features_cache.pkl"
if os.path.exists(CACHE_FILE):
    print("Loading cached features")
    with open(CACHE_FILE, 'rb') as f:
        train_features, test_features, y_train, y_test = pickle.load(f)
    y_train_arr = y_train
    y_test_arr = y_test
else:
    def extract_features(model, loader,desc="Extracting"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        features_list = []
        labels_list = []
        print(f"Extracting features on {device}")
        with torch.no_grad():
            for batch_images, batch_labels in tqdm(loader, desc="Extracting"):
                batch_images = batch_images.to(device,non_blocking=True)
                with autocast(enabled=torch.cuda.is_available()):
                    features = model(batch_images)
                features_list.append(features.cpu())
                labels_list.append(batch_labels)

        all_features = torch.cat(features_list, dim=0)
        all_labels = torch.cat([torch.tensor(l) if not isinstance(l, torch.Tensor) else l for l in labels_list], dim=0)

        return all_features.numpy(), all_labels.numpy()

    train_features, y_train_arr = extract_features(feature_extractor, train_loader)
    test_features, y_test_arr = extract_features(feature_extractor, test_loader)
    print("Feature vector shape:", train_features.shape)
    print("Train labels shape:", y_train_arr.shape)
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump((train_features, test_features, y_train_arr, y_test_arr), f)
    print(f"Features cached to {CACHE_FILE}")

print(f"Train features: {train_features.shape}")
print(f"Test features: {test_features.shape}")
print(f"Train labels: {y_train_arr.shape}")
print(f"Test labels: {y_test_arr.shape}")
print(f"Unique classes: {len(np.unique(y_train_arr))}")





pipeline = Pipeline([
    ('pca', PCA(n_components=1, svd_solver='randomized')), 
    ('svm', SVC(kernel='rbf', probability=False, cache_size=4000)) 
])
param_grid = {'svm__C': [1, 10, 100],'svm__gamma': ['scale', 0.001, 0.01]}
svm_search = HalvingGridSearchCV(pipeline,param_grid,cv=3,factor=3,resource='n_samples',min_resources=1300,n_jobs=-1,verbose=2)
print("Starting training")
svm_search.fit(train_features, y_train_arr)
print(f"Best parameters: {svm_search.best_params_}")
final_svm_model = svm_search.best_estimator_
final_svm_model.set_params(svm__probability=True)
final_svm_model.fit(train_features, y_train_arr)
y_pred = final_svm_model.predict(test_features)
accuracy = accuracy_score(y_test_arr, y_pred)
cm = confusion_matrix(y_test_arr, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title(f'Confusion Matrix - Accuracy: {accuracy:.4f}')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()





joblib.dump(svm_search, "svm_plant_disease_model.pkl")
torch.save(feature_extractor.state_dict(), "feature_extractor.pth")

if __name__ == "__main__":
    show_sample_images()
    print("\n" + "="*50)
    print("SCRIPT EXECUTION COMPLETE!")
    print("="*50)
    print(f"Final Accuracy: {accuracy:.2f}")
    print(f"Models saved successfully")