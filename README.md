 
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


## 🔧 Installation
To set up the environment, run:
```bash
pip install ultralytics opencv-python numpy torch torchvision tqdm
```

# 📊 YOLOv8 Classification Results

---

## 🏗 Model Training Settings
We trained **YOLOv8s-cls** (small classification model) with the following settings:

| Parameter  | Value  | Description |
|------------|--------|-------------|
| `epochs`   | **300**  | Number of training epochs |
| `imgsz`    | **224**  | Image size for input |
| `batch`    | **64**  | Batch size for training |
| `workers`  | **2**   | Number of CPU workers for data loading |
| `optimizer` | **Adam** | Optimizer used for weight updates |
| `lr0`      | **0.0001** | Initial learning rate |
| `dropout`  | **0.2** | Dropout rate to prevent overfitting |
| `cos_lr`   | **True** | Use cosine learning rate schedule |
| `hsv_h`    | **0.015** | Hue augmentation parameter |
| `hsv_s`    | **0.7**  | Saturation augmentation parameter |
| `hsv_v`    | **0.4**  | Brightness augmentation parameter |
| `fliplr`   | **0.5**  | Probability of horizontal flip |
| `device`   | **cuda:0** | Force GPU usage |

## 📊 Class Distribution in Train, Validation, and Test Sets

![Class Distribution](images/class_distribution.png)  

The bar chart above represents the distribution of images across different classes in the training, validation, and test datasets. This helps in understanding class imbalance and dataset composition:

- **Train (Blue)**: Majority of images are allocated to training to allow the model to learn effectively.
- **Validation (Orange)**: A smaller set is used to tune hyperparameters and monitor overfitting.
- **Test (Green)**: A separate set for evaluating the final model performance.

This visualization ensures that all 11 classes are well represented across all splits and helps in improving model generalization.

---
## 🖥️ Training Log Screenshot

Here is a snapshot of the training setup and model structure:

![Training Log](images/training_log.png)


### 🎯 **Final Model Performance**
- **Top-1 Accuracy**: **62.5%**
- **Top-5 Accuracy**: **91.0%**
- **Test Set Size**: **542 images**
- **Total Classes**: **11**




