"""
--------------------------------------------------
YOLOv8 Chest X-Ray Disease Classification  
Developer: Usama Yasir Khan  
Script: Convert Bounding Box Labels to Classification Labels  
--------------------------------------------------
"""

import os
import json
from tqdm import tqdm

# ✅ Dataset path
dataset_path = "/kaggle/working/yolo_classification"
train_folder = os.path.join(dataset_path, "train")
valid_folder = os.path.join(dataset_path, "valid")
test_folder = os.path.join(dataset_path, "test")
annotation_file = "/kaggle/input/chestxdet10dataset/train.json"

# ✅ Class mapping
category_to_index = {
    "Consolidation": 0, "Pneumothorax": 1, "Emphysema": 2, "Calcification": 3,
    "Nodule": 4, "Mass": 5, "Fracture": 6, "Effusion": 7, "Atelectasis": 8, "Fibrosis": 9,
    "No_Finding": 10
}

# ✅ Read annotation file
with open(annotation_file, "r") as f:
    annotations = json.load(f)

# ✅ Convert annotations
for ann in tqdm(annotations, desc="Converting Labels"):
    image_name = ann["file_name"]
    class_labels = set(ann.get("syms", ["No_Finding"]))
    main_class = next(iter(class_labels))  

    if main_class not in category_to_index:
        print(f"⚠️ Unknown class '{main_class}' for {image_name}")
        continue

    class_id = category_to_index[main_class]
    label_path = os.path.join(dataset_path, image_name.replace(".png", ".txt"))

    with open(label_path, "w") as f:
        f.write(f"{class_id}\n")

print("✅ Label conversion complete!")
