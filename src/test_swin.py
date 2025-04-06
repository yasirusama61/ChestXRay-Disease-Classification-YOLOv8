import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
from PIL import Image
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import classification_report
from tqdm import tqdm

# === Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 11
IMG_SIZE = 224
BATCH_SIZE = 32
TEST_IMG_DIR = "/kaggle/input/chestxdet10dataset/test_data/test_data"
TEST_ANN_PATH = "/kaggle/input/chestxdet10dataset/test.json"
MODEL_PATH = "/kaggle/working/swin_chestxdet10_best.pth"

# === Category mapping
category_to_index = {
    'Consolidation': 0, 'Pneumothorax': 1, 'Emphysema': 2, 'Calcification': 3,
    'Nodule': 4, 'Mass': 5, 'Fracture': 6, 'Effusion': 7,
    'Atelectasis': 8, 'Fibrosis': 9, 'No Finding': 10
}
index_to_category = {v: k for k, v in category_to_index.items()}
mlb = MultiLabelBinarizer(classes=list(range(NUM_CLASSES)))

# === Dataset
class ChestXDet10Dataset(torch.utils.data.Dataset):
    def __init__(self, annotation_path, image_dir, transform=None):
        with open(annotation_path, 'r') as f:
            annotations = json.load(f)
        self.image_dir = image_dir
        self.transform = transform
        self.data = []
         for sample in annotations:
            labels = sample['syms']
            if not labels:
                label_indices = [category_to_index['No Finding']]
            else:
                label_indices = [category_to_index[l] for l in labels]
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

# === Transforms
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485], [0.229])
])

# === Load test data
test_dataset = ChestXDet10Dataset(TEST_ANN_PATH, TEST_IMG_DIR, transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# === Load model
import timm
model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# === Evaluation
all_labels, all_preds = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        probs = torch.sigmoid(outputs).cpu().numpy()
        all_preds.extend(probs)
        all_labels.extend(labels.numpy())

# Binarize predictions with threshold = 0.5
y_pred_bin = (np.array(all_preds) >= 0.5).astype(int)
y_true = np.array(all_labels)

# Classification report and AUC
report = classification_report(y_true, y_pred_bin, target_names=[index_to_category[i] for i in range(NUM_CLASSES)], output_dict=True)
auc = roc_auc_score(y_true, all_preds, average=None)


report_df = pd.DataFrame(report).transpose()
report_df["AUC"] = list(auc) + [np.nan] * (len(report_df) - len(auc))

# === Load model
model.load_state_dict(torch.load("/kaggle/working/swin_chestxdet10_best.pth", map_location=DEVICE))
model.eval()

y_true = []
y_probs = []

with torch.no_grad():
    for images, labels in tqdm(test_loader):
        images = images.to(DEVICE)
        outputs = model(images)

        probs = torch.sigmoid(outputs).cpu().numpy()
        y_probs.extend(probs)

        y_true.extend(labels.numpy())

y_true = np.array(y_true)
y_probs = np.array(y_probs)

# === Save for Youden evaluation
np.save("/kaggle/working/y_true.npy", y_true)
np.save("/kaggle/working/y_probs.npy", y_probs)

# === Optional: basic threshold=0.5 report
y_pred = (y_probs > 0.5).astype(int)
report = classification_report(y_true, y_pred, target_names=category_to_index.keys(), zero_division=0)
print(report)