import os
import json
import random
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_score, recall_score, roc_curve
import timm

# === Config
IMG_SIZE = 224
BATCH_SIZE = 64
NUM_CLASSES = 11
EPOCHS = 100
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
standard_aug = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485], [0.229])
])

val_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485], [0.229])
])

# === Early Stopping
class EarlyStopping:
    def __init__(self, patience=15, delta=1e-4):
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
class MergedChestDataset(Dataset):
    def __init__(self, annotation_file, image_dir, transform=standard_aug):
        self.image_dir = image_dir
        self.transform = transform
        with open(annotation_file, 'r') as f:
            raw_data = json.load(f)
        self.samples = [(item['file_name'], item['syms']) for item in raw_data]
        self.mlb = MultiLabelBinarizer(classes=list(range(NUM_CLASSES)))
        self.mlb.fit([list(range(NUM_CLASSES))])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, labels = self.samples[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        label_indices = [category_to_index[l] for l in labels] if labels else [category_to_index['No Finding']]
        label_vector = self.mlb.transform([label_indices])[0]
        return image, torch.FloatTensor(label_vector)

# === Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return loss.mean() if self.reduction == 'mean' else loss.sum()

# === CutMix Function
def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(x.size()[0])
    y_a, y_b = y, y[rand_index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
    y = torch.max(y_a, y_b)  # multilabel logic
    return x, y

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

# === Weighted Sampler
def get_weighted_sampler(dataset):
    label_freq = np.zeros(NUM_CLASSES)
    for _, labels in dataset:
        label_freq += labels.numpy()

    weights = 1.0 / (label_freq + 1e-6)
    sample_weights = []
    for _, labels in dataset:
        sample_weights.append(weights[labels.numpy().astype(bool)].mean())
    return WeightedRandomSampler(sample_weights, len(sample_weights))

# === Metrics

def compute_classwise_thresholds(y_true, y_probs):
    thresholds = []
    for i in range(y_true.shape[1]):
        fpr, tpr, thr = roc_curve(y_true[:, i], y_probs[:, i])
        youden = tpr - fpr
        best = thr[np.argmax(youden)] if len(thr) > 0 else 0.5
        thresholds.append(best)
    return thresholds

def compute_metrics(y_true, y_probs, thresholds):
    y_pred_bin = (y_probs >= thresholds).astype(int)
    metrics = {}
    for i in range(NUM_CLASSES):
        try:
            p = precision_score(y_true[:, i], y_pred_bin[:, i], zero_division=0)
            r = recall_score(y_true[:, i], y_pred_bin[:, i], zero_division=0)
        except:
            p, r = np.nan, np.nan
        metrics[index_to_category[i]] = {"precision": p, "recall": r}
    return metrics

# === Paths
DATA_DIR = "./merged_train_images"
ANNOT_FILE = "./merged_annotations.json"

# === Data Loading
full_dataset = MergedChestDataset(ANNOT_FILE, DATA_DIR)
val_size = int(0.2 * len(full_dataset))
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
val_dataset.dataset.transform = val_tfms

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=get_weighted_sampler(train_dataset))
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# === Model
model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=NUM_CLASSES).to(DEVICE)
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
            if random.random() < 0.5:
                images, labels = cutmix_data(images, labels)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        
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

        print(f"\n🔁 Epoch {epoch+1} | LR: {scheduler.get_last_lr()[0]:.6f} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")
        for cls, m in metrics.items():
            print(f"  - {cls:14s} | Precision: {m['precision']:.4f} | Recall: {m['recall']:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "swin_balanced_best.pth")

        early_stopper(val_loss)
        if early_stopper.early_stop:
            print("\n🛑 Early stopping triggered.")
            break

# === Run Training
train()
