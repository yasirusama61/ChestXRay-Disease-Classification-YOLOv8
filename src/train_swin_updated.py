import os
import json
import random
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_score, recall_score, roc_curve
import timm

# === Config
IMG_SIZE = 224
BATCH_SIZE = 64
NUM_CLASSES = 11
EPOCHS = 200
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Class Mapping
category_to_index = {
    'Consolidation': 0, 'Pneumothorax': 1, 'Emphysema': 2, 'Calcification': 3,
    'Nodule': 4, 'Mass': 5, 'Fracture': 6, 'Effusion': 7,
    'Atelectasis': 8, 'Fibrosis': 9, 'No Finding': 10
}
index_to_category = {v: k for k, v in category_to_index.items()}
minority_classes = ['Pneumothorax', 'Calcification', 'Fracture', 'Fibrosis', 'Mass', 'Nodule', 'Atelectasis']

# === Transforms
strong_aug = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=10, shear=10),
    transforms.ToTensor(),
    transforms.Normalize([0.485], [0.229])
])

standard_aug = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485], [0.229])
])

val_tfms = standard_aug

# === Early Stopping
class EarlyStopping:
    def __init__(self, patience=10, delta=1e-4):
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# === Dataset
class ChestXDet10Dataset(Dataset):
    def __init__(self, annotation_file, image_dir):
        self.image_dir = image_dir

        with open(annotation_file, 'r') as f:
            raw_data = json.load(f)
            self.annotations = {item['file_name']: item['syms'] for item in raw_data}

        self.img_labels = []
        for img_name, labels in self.annotations.items():
            label_indices = [category_to_index[l] for l in labels] if labels else [category_to_index['No Finding']]
            self.img_labels.append((img_name, label_indices))
            if any(index_to_category[i] in minority_classes for i in label_indices):
                for _ in range(3):
                    self.img_labels.append((img_name, label_indices))

        self.mlb = MultiLabelBinarizer(classes=list(range(NUM_CLASSES)))
        self.mlb.fit([list(range(NUM_CLASSES))])

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name, label_indices = self.img_labels[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        original_labels = [index_to_category[i] for i in label_indices]
        transform = strong_aug if any(cls in minority_classes for cls in original_labels) else standard_aug
        image = transform(image)

        label_vector = self.mlb.transform([label_indices])[0]
        return image, torch.FloatTensor(label_vector)

# === Model Training
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()

# === Training Utilities

def compute_classwise_thresholds(y_true, y_probs):
    thresholds = []
    for i in range(y_true.shape[1]):
        fpr, tpr, thr = roc_curve(y_true[:, i], y_probs[:, i])
        youden = tpr - fpr
        best = thr[np.argmax(youden)] if len(thr) > 0 else 0.5
        thresholds.append(best)
    return thresholds

def compute_metrics(y_true, y_probs, thresholds=None):
    if thresholds is None:
        thresholds = [0.5] * NUM_CLASSES
    y_pred_bin = (y_probs >= thresholds).astype(int)
    metrics = {}
    for i in range(NUM_CLASSES):
        try:
            precision = precision_score(y_true[:, i], y_pred_bin[:, i], zero_division=0)
            recall = recall_score(y_true[:, i], y_pred_bin[:, i], zero_division=0)
        except:
            precision = np.nan
            recall = np.nan
        metrics[index_to_category[i]] = {"precision": precision, "recall": recall}
    return metrics

# === Paths
DATA_DIR = "dataset/train_data/train-old"
ANNOTATION_FILE = "dataset/train.json"

# === Dataset and Dataloaders
dataset = ChestXDet10Dataset(ANNOTATION_FILE, DATA_DIR)
val_size = int(0.2 * len(dataset))
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
val_dataset.dataset.transform = val_tfms

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# === Model, Loss, Optimizer
model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=NUM_CLASSES)
model.to(DEVICE)
criterion = FocalLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# === Training Loop
def train():
    best_val_loss = float("inf")
    early_stopper = EarlyStopping(patience=20)
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        # === Validation ===
        model.eval()
        val_loss = 0
        all_labels, all_probs = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                all_labels.append(labels.cpu().numpy())
                all_probs.append(torch.sigmoid(outputs).cpu().numpy())

        y_true = np.concatenate(all_labels, axis=0)
        y_probs = np.concatenate(all_probs, axis=0)
        thresholds = compute_classwise_thresholds(y_true, y_probs)
        metrics = compute_metrics(y_true, y_probs, thresholds)

        print(f"\n🔁 Epoch {epoch+1}/{EPOCHS} | LR: {scheduler.get_last_lr()[0]:.6f} | Train Loss: {train_loss / len(train_loader):.4f} | Val Loss: {val_loss / len(val_loader):.4f}")
        print("📊 Precision & Recall per class:")
        for cls, m in metrics.items():
            print(f"  - {cls:14s} | Precision: {m['precision']:.4f} | Recall: {m['recall']:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "swin_chestxdet10_best.pth")

        early_stopper(val_loss)
        if early_stopper.early_stop:
            print("🛑 Early stopping triggered. Training halted.")
            break

# Run training
train()