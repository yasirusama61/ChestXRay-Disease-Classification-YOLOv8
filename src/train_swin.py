import os, json, torch, timm
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# === Config
IMG_SIZE = 224
BATCH_SIZE = 32
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

# === Custom Dataset
class ChestXDet10Dataset(Dataset):
    def __init__(self, annotation_file, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        with open(annotation_file, 'r') as f:
            raw_data = json.load(f)
            self.annotations = {item['file_name']: item['syms'] for item in raw_data}

        self.img_labels = []
        for img_name, labels in self.annotations.items():
            label_indices = [category_to_index[l] for l in labels] if labels else [category_to_index['No Finding']]
            self.img_labels.append((img_name, label_indices))

        self.mlb = MultiLabelBinarizer(classes=list(range(NUM_CLASSES)))
        self.mlb.fit([list(range(NUM_CLASSES))])

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name, label_indices = self.img_labels[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label_vector = self.mlb.transform([label_indices])[0]
        return image, torch.FloatTensor(label_vector)

# === Transforms
train_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485], [0.229])
])
val_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485], [0.229])
])
# === Dataset Paths
DATA_DIR = "/kaggle/input/chestxdet10dataset/train_data/train-old"
ANNOTATION_FILE = "/kaggle/input/chestxdet10dataset/train.json"

# === Dataloaders
dataset = ChestXDet10Dataset(ANNOTATION_FILE, DATA_DIR, transform=train_tfms)
val_size = int(0.2 * len(dataset))
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
val_dataset.dataset.transform = val_tfms

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

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
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()

# === Model, Loss, Optimizer
model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=NUM_CLASSES)
model.to(DEVICE)

criterion = FocalLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# === AUROC per class
def compute_auroc(y_true, y_probs):
    scores = {}
    for i in range(NUM_CLASSES):
        try:
            scores[index_to_category[i]] = roc_auc_score(y_true[:, i], y_probs[:, i])
        except:
            scores[index_to_category[i]] = np.nan
    return scores

# === Training
def train():
    best_val_loss = float("inf")
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

        # === Validation
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
        aurocs = compute_auroc(y_true, y_probs)

        print(f"🔁 Epoch {epoch+1}/{EPOCHS} | LR: {scheduler.get_last_lr()[0]:.6f} | Train Loss: {train_loss / len(train_loader):.4f} | Val Loss: {val_loss / len(val_loader):.4f}")
        print("📊 AUROC per class:")
        for cls, auc in aurocs.items():
            print(f"  - {cls:14s}: {auc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "swin_chestxdet10_best.pth")

train()