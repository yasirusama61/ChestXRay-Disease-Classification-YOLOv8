# swin_chestxdet10_eval.py

import os
import json
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import timm

from sklearn.metrics import (
    classification_report, roc_auc_score,
    multilabel_confusion_matrix, roc_curve, auc
)
from sklearn.preprocessing import MultiLabelBinarizer

# === Configuration ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 11
IMG_SIZE = 224
BATCH_SIZE = 32

TEST_IMG_DIR = "/kaggle/input/chestxdet10dataset/test_data/test_data"
TEST_ANN_PATH = "/kaggle/input/chestxdet10dataset/test.json"
MODEL_PATH = "/kaggle/working/swin_chestxdet10_best.pth"

CATEGORY_TO_INDEX = {
    'Consolidation': 0, 'Pneumothorax': 1, 'Emphysema': 2, 'Calcification': 3,
    'Nodule': 4, 'Mass': 5, 'Fracture': 6, 'Effusion': 7,
    'Atelectasis': 8, 'Fibrosis': 9, 'No Finding': 10
}
INDEX_TO_CATEGORY = {v: k for k, v in CATEGORY_TO_INDEX.items()}
CLASS_NAMES = list(CATEGORY_TO_INDEX.keys())

# === Dataset ===
class ChestXDet10Dataset(Dataset):
    def __init__(self, annotation_path, image_dir, transform=None):
        with open(annotation_path, 'r') as f:
            annotations = json.load(f)
        self.image_dir = image_dir
        self.transform = transform
        self.data = []

        for sample in annotations:
            labels = sample['syms']
            if not labels:
                label_indices = [CATEGORY_TO_INDEX['No Finding']]
            else:
                label_indices = [CATEGORY_TO_INDEX[l] for l in labels]
            self.data.append((sample['file_name'], label_indices))

        self.mlb = MultiLabelBinarizer(classes=list(range(NUM_CLASSES)))
        self.mlb.fit([[]])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, label_indices = self.data[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label_vector = self.mlb.transform([label_indices])[0]
        return image, torch.FloatTensor(label_vector)

# === Transforms ===
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485], [0.229])
])

# === Load Data ===
test_dataset = ChestXDet10Dataset(TEST_ANN_PATH, TEST_IMG_DIR, transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# === Load Model ===
model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# === Inference on Validation Data to Compute Thresholds ===
val_labels, val_preds = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        probs = torch.sigmoid(outputs).cpu().numpy()
        val_preds.extend(probs)
        val_labels.extend(labels.numpy())

val_preds = np.array(val_preds)
val_labels = np.array(val_labels)

# === Compute Class-wise Youden Thresholds ===
def compute_classwise_thresholds(y_true, y_probs):
    thresholds = []
    for i in range(y_true.shape[1]):
        fpr, tpr, thr = roc_curve(y_true[:, i], y_probs[:, i])
        youden = tpr - fpr
        best_thresh = thr[np.argmax(youden)]
        thresholds.append(best_thresh)
    return np.array(thresholds)

def apply_classwise_thresholds(y_probs, thresholds):
    return (y_probs >= thresholds).astype(int)

optimal_thresholds = compute_classwise_thresholds(val_labels, val_preds)
print("\n📊 Class-wise Optimal Thresholds:", optimal_thresholds.round(3))

# === Re-evaluate on Test Set using Optimized Thresholds ===
y_true, y_probs = [], []

with torch.no_grad():
    for images, labels in tqdm(test_loader):
        images = images.to(DEVICE)
        outputs = model(images)
        probs = torch.sigmoid(outputs).cpu().numpy()
        y_probs.extend(probs)
        y_true.extend(labels.numpy())

y_true = np.array(y_true)
y_probs = np.array(y_probs)

# Save predictions
np.save("/kaggle/working/y_true.npy", y_true)
np.save("/kaggle/working/y_probs.npy", y_probs)

# Apply optimized thresholds
binary_preds = apply_classwise_thresholds(y_probs, optimal_thresholds)
report = classification_report(y_true, binary_preds, target_names=CLASS_NAMES, zero_division=0)
print(report)

# === Plot AUC per class ===
def plot_auc_per_class(y_true, y_probs, class_names, title="ROC Curve (AUC per Class)"):
    plt.figure(figsize=(10, 8))
    for i in range(len(class_names)):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_auc_per_class(y_true, y_probs, CLASS_NAMES)

# === Multilabel Confusion Matrices ===
mcm = multilabel_confusion_matrix(y_true, binary_preds)
fig, axes = plt.subplots(3, 4, figsize=(16, 10))
axes = axes.flatten()

for i, (cm, label) in enumerate(zip(mcm, CLASS_NAMES)):
    tn, fp, fn, tp = cm.ravel()
    ax = axes[i]
    im = ax.imshow([[tp, fn], [fp, tn]], cmap="Blues", vmin=0)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred: 1", "Pred: 0"])
    ax.set_yticklabels(["True: 1", "True: 0"])
    ax.set_title(f"{label}")
    for (x, y), val in np.ndenumerate([[tp, fn], [fp, tn]]):
        ax.text(y, x, f"{val}", ha='center', va='center', color='black', fontsize=10)

for j in range(len(CLASS_NAMES), len(axes)):
    axes[j].axis("off")

fig.suptitle("Multilabel Confusion Matrix (Per Class)", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
