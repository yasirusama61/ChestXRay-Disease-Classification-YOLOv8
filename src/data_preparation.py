import os
import shutil
import json

# Define dataset paths
dataset_path = "/kaggle/input/chestxdet10dataset"
train_image_folder = os.path.join(dataset_path, "train_data", "train-old")
test_image_folder = os.path.join(dataset_path, "test_data", "test_data")
train_annotation_file = os.path.join(dataset_path, "train.json")
test_annotation_file = os.path.join(dataset_path, "test.json")
output_folder = "/kaggle/working/yolo_classification"

# Category mapping based on dataset description
category_to_index = {
    'Consolidation': 0, 'Pneumothorax': 1, 'Emphysema': 2, 'Calcification': 3,
    'Nodule': 4, 'Mass': 5, 'Fracture': 6, 'Effusion': 7, 'Atelectasis': 8, 'Fibrosis': 9,
    'No_Finding': 10
}

# Create directories for train, valid, and test
for split in ["train", "valid", "test"]:
    os.makedirs(os.path.join(output_folder, split), exist_ok=True)
    for class_name in [
        "Consolidation", "Pneumothorax", "Emphysema", "Calcification",
        "Nodule", "Mass", "Fracture", "Effusion", "Atelectasis", "Fibrosis", "No_Finding"
    ]:
        os.makedirs(os.path.join(output_folder, split, class_name), exist_ok=True)

# Function to organize images into class folders
def organize_images(image_folder, annotation_file, split):
    with open(annotation_file, "r") as f:
        annotations = json.load(f)

    for ann in annotations:
        image_name = ann["file_name"]
        image_path = os.path.join(image_folder, image_name)
        if not os.path.exists(image_path):
            print(f"⚠️ Image {image_name} not found. Skipping.")
            continue
        
        class_labels = set(ann["syms"]) if "syms" in ann and ann["syms"] else {"No_Finding"}
        main_class = next(iter(class_labels))
        if main_class not in category_to_index:
            print(f"⚠️ Unknown class '{main_class}' in {image_name}. Skipping.")
            continue
        
        dest_folder = os.path.join(output_folder, split, main_class)
        shutil.copy(image_path, os.path.join(dest_folder, image_name))

# Process train and test datasets
organize_images(train_image_folder, train_annotation_file, "train")
organize_images(test_image_folder, test_annotation_file, "test")

print("✅ Data preparation completed. Images sorted into class folders.")