 
# YOLOv8 Chest X-Ray Disease Classification

## 🚀 Project Overview
This project implements **YOLOv8** for classifying chest X-ray images into 11 disease categories. The model is trained on the **ChestXDet10** dataset to automate medical diagnosis using deep learning.

## 📂 Dataset
We use the **ChestXDet10 Dataset** containing annotated X-ray images. The dataset includes 11 classes:

1. **Consolidation**  
2. **Pneumothorax**  
3. **Emphysema**  
4. **Calcification**  
5. **Nodule**  
6. **Mass**  
7. **Fracture**  
8. **Effusion**  
9. **Atelectasis**  
10. **Fibrosis**  
11. **No Finding** (normal cases)  

## 🏗️ Project Structure

- **`data/`**: Contains the dataset (`train/`, `valid/`, `test/`) for classification.
- **`scripts/`**: Python scripts for data preparation, label conversion, and training.
  - `data_preparation.py` → Organizes dataset into `train/`, `valid/`, `test/`
  - `convert_labels.py` → Converts YOLO bounding box annotations to classification labels
  - `train.py` → Trains YOLOv8 classification model
- **`models/`**: Saved trained YOLOv8 models (`best.pt`).
- **`results/`**: Evaluation results, including accuracy reports and confusion matrices.
- **`inference/`**: Script for testing trained models on new images.
  - `inference.py` → Runs inference on new X-ray images
- **`requirements.txt`**: Python dependencies for the project.
- **`.gitignore`**: Ignored files for version control (e.g., large datasets, logs).
- **`README.md`**: Project documentation.


