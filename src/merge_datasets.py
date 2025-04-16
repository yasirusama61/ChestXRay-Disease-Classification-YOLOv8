import os
import json
import shutil

# === Mappings: external dataset → label
source_label_map = {
    "siim": "Pneumothorax",
    "vinbigdata": "Nodule",
    "pneumothorax": "Atelectasis"
}

# === Paths
base_dir = "/kaggle/input"
original_data_dir = os.path.join(base_dir, "chestxdet10dataset", "train_data", "train-old")
annotation_file = os.path.join(base_dir, "chestxdet10dataset", "train.json")
output_img_dir = "./merged_train_images"
output_annot_path = "./merged_annotations.json"

# === External dataset folders (can be lists if multiple sources)
external_sources = {
    "siim": [
        os.path.join(base_dir, "pneumothorax-chest-xray-images-and-masks", "siim-acr-pneumothorax", "png_images")
    ],
    "vinbigdata": [
        os.path.join(base_dir, "vinbigdata-chest-xray-original-png", "train"),
        os.path.join(base_dir, "vinbigdata-chest-xray-original-png", "test")
    ],
    "pneumothorax": [
        os.path.join(base_dir, "siim-acr-pneumothorax", "siim-acr-pneumothorax-NEW", "training", "input"),
        os.path.join(base_dir, "siim-acr-pneumothorax", "siim-acr-pneumothorax-NEW", "testing", "input")
    ]
}

# === Prepare output directory
os.makedirs(output_img_dir, exist_ok=True)

# === Step 1: Load & copy original ChestXDet10 images
with open(annotation_file, "r") as f:
    original_data = json.load(f)

merged_annotations = []
for item in original_data:
    src_path = os.path.join(original_data_dir, item['file_name'])
    dst_path = os.path.join(output_img_dir, item['file_name'])
    if os.path.exists(src_path):
        shutil.copy(src_path, dst_path)
        merged_annotations.append({
            "file_name": item['file_name'],
            "syms": item['syms'] if item['syms'] else ["No Finding"]
        })

# === Step 2: Add External Samples (limit to 500 per source)
for source_name, folders in external_sources.items():
    for folder in folders:
        files = sorted([f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg'))])[:500]
        for fname in files:
            new_name = f"{source_name}_{fname}"
            shutil.copy(os.path.join(folder, fname), os.path.join(output_img_dir, new_name))
            merged_annotations.append({
                "file_name": new_name,
                "syms": [source_label_map[source_name]]
            })

# === Step 3: Save final merged annotation JSON
with open(output_annot_path, "w") as f:
    json.dump(merged_annotations, f, indent=2)

print(f"✅ Merged {len(merged_annotations)} annotations into {output_annot_path}")
