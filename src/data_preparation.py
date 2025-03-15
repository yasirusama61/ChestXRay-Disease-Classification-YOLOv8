"""
--------------------------------------------------
YOLOv8 Chest X-Ray Disease Classification  
Developer: Usama Yasir Khan  
Script: Data Preparation  
--------------------------------------------------
"""

import os
import json
import shutil
from tqdm import tqdm

# ✅ Dataset path
dataset_path = "/kaggle/input/chestxdet10dataset"
train_image_folder = os.path.join(dataset_path, "train_data", "train-old")
test_image_folder = os.path.join(dataset_path, "test_data", "test_data")
train_annotation_file = os.path.join(dataset_path, "train.json")
test_annotation_file = os.path.join(dataset_path, "test.json")

# ✅ YOLO classification dataset structure
classification_dataset_path = "/kaggle/working/yolo_classification"
train_folder = os.path.join(classification_dataset_path, "train")
valid_folder = os.path.join(classification_dataset_path, "valid")
test_folder = os.path.join(classification_dataset_path, "test")

# ✅ Class mapping
category_to_index = {
    "Consolidation": 0, "Pneumothorax": 1, "Emphysema": 2, "Calcification": 3,
    "Nodule": 4, "Mass": 5, "Fracture": 6, "Effusion": 7, "Atelectasis": 8, "Fibrosis": 9,
    "No_Finding": 10
}

# ✅ Create folders
for folder in [train_folder, valid_folder, test_folder]:
    for class_name in category_to_index.keys():
        os.makedirs(os.path.join(folder, class_name), exist_ok=True)

# ✅ Function to distribute images
def distribute_images(image_folder, annotation_file, output_folder):
    with open(annotation_file, "r") as f:
        annotations = json.load(f)

    for ann in tqdm(annotations, desc=f"Processing {output_folder}"):
        image_name = ann["file_name"]
        image_path = os.path.join(image_folder, image_name)

        if not os.path.exists(image_path):
            print(f"⚠️ Skipping {image_name} (not found)")
            continue

        class_labels = set(ann.get("syms", ["No_Finding"]))
        main_class = next(iter(class_labels))  

        if main_class not in category_to_index:
            print(f"⚠️ Unknown class '{main_class}' for {image_name}")
            continue

        dest_folder = os.path.join(output_folder, main_class)
        shutil.copy(image_path, os.path.join(dest_folder, image_name))

# ✅ Distribute images
distribute_images(train_image_folder, train_annotation_file, train_folder)
distribute_images(test_image_folder, test_annotation_file, test_folder)

print("✅ Dataset prepared successfully!")
