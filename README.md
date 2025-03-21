 
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

## 📸 Sample Training Batch

Below is a sample batch of images from the **training dataset**, showing various chest X-ray images with diverse conditions:

![Training Batch](images/train_batch0.jpg)

## 🏥 Chest X-ray Data Augmentation Example

The image below showcases **a batch of training images** after applying **data augmentations**, including:
- **Random Cropping**
- **Brightness Adjustments**
- **Random Rotations**
- **Black Box Occlusions** (to simulate missing data)
- **Contrast Adjustments**

<p align="center">
  <img src="images/train_batch_aug.jpg" alt="Sample Training Batch" width="600"/>
</p>

---

### 🛠️ **Data Processing Pipeline**
1. **Preprocessing:** Convert grayscale X-ray images to a standard resolution.
2. **Augmentation:** Apply transformations to increase model robustness.
3. **Label Encoding:** Convert disease labels into YOLO classification format.
4. **Training:** Fine-tune **YOLOv8** on the preprocessed dataset.

📌 _These augmentations help the model generalize better and improve classification accuracy!_ 🚀

## 📈 Training Metrics and Loss Curves

The following plots represent the training loss curves, classification loss, and performance metrics during the training process:

![Training Metrics](images/results.png)

## 📉 Loss Curve Analysis

The training and validation loss curves provide insights into the model's learning process.

### 🔍 Observations:
- **Train Loss:** Gradually decreases, showing stable learning.
- **Validation Loss:** Initially drops but plateaus, indicating limited further learning.
- **Precision & Recall:** Precision increases over epochs but fluctuates, while recall stabilizes between **0.5 - 0.7**.
- **mAP Scores:** Consistently improving, confirming better predictions.

### ⚠️ Potential Issues:
- **Validation loss plateauing** → Model may require fine-tuning (learning rate decay, weight decay).
- **Fluctuations in precision & recall** → Some classes might be harder to classify due to **imbalance**.
- **Slight Drop in Top-1 Accuracy** compared to previous runs.

### 🔧 Recommendations:
1. **Use Cosine Learning Rate Decay** (`cos_lr=True`) to improve convergence.
2. **Enhance Data Augmentation** to handle class imbalance.
3. **Increase Dropout Regularization** (`dropout=0.3`) to prevent overfitting.
4. **Try a Larger Model** (`YOLOv8m-cls`) for better feature extraction.

🚀 *Next Step:* Re-run training with adjustments & monitor validation loss trends.

### 🎯 **Final Model Performance**
- **Top-1 Accuracy**: **62.5%**
- **Top-5 Accuracy**: **91.0%**
- **Test Set Size**: **542 images**
- **Total Classes**: **11**

## 🛠 Solving Multi-Label Per Image Classification in YOLO  

### **🚀 The Problem**  
YOLO is originally designed for **single-class object detection**, meaning each detected object (bounding box) is assigned only **one label**. However, in **medical imaging**, an X-ray or MRI can contain **multiple conditions simultaneously**.  

#### **🛑 Issues Encountered:**
- Training YOLO as a **classification model** (subfolder-based approach) resulted in **poor performance**, the top 1 accuracy is just 60%.
- Directly using **YOLO detection** led to **low precision** due to **incorrect class assignments**.
- **No built-in support for multi-label classification** per image.

---

### **✅ Our Solution: Treating Multi-Label Classification as Object Detection**
Since YOLO expects **object detection annotations**, we **tricked YOLO into multi-label learning** by:
1. **Using a bounding box that covers the entire image** for each label.
2. **Allowing multiple bounding boxes (one per disease) per image**.
3. **Training YOLO in detection mode** instead of classification mode.

**📌 Example YOLO Annotation (`image.txt`) for an X-ray with "Effusion" and "Atelectasis":**
```txt
7 0.5 0.5 1.0 1.0  # Effusion (Class 7)
8 0.5 0.5 1.0 1.0  # Atelectasis (Class 8)
```
## 📊 Training Results
Training was conducted on the **NIH dataset** with **multi-label YOLOv8 detection**.

- **Total Training Time:** 0.917 hours (100 epochs)
- **Model:** YOLOv8x
- **Dataset:** NIH Chest X-ray (14 classes)
- **Optimizer:** Adam
- **Training Directory:** `/kaggle/working/yolo_multilabel_results/train_multilabel_v12/`

### **📈 Final Validation Metrics**
| **Metric** | **Value** |
|------------|----------|
| **mAP@50** | **0.488** |
| **mAP@50-95** | **0.488** |
| **Precision (P)** | **0.44** |
| **Recall (R)** | **0.535** |

### **📌 Class-wise Performance**
| **Class** | **Precision (P)** | **Recall (R)** | **mAP@50** | **mAP@50-95** |
|-----------|-----------------|-----------------|------------|--------------|
| **Consolidation** | 0.648 | 0.883 | 0.839 | 0.839 |
| **Pneumothorax** | 0.209 | 0.25  | 0.205 | 0.205 |
| **Emphysema** | 0.647 | 0.654 | 0.618 | 0.618 |
| **Calcification** | 0.134 | 0.381 | 0.189 | 0.189 |
| **Nodule** | 0.309 | 0.215 | 0.282 | 0.282 |
| **Mass** | 0.287 | 0.364 | 0.243 | 0.243 |
| **Fracture** | 0.414 | 0.414 | 0.492 | 0.492 |
| **Effusion** | 0.736 | 0.873 | 0.868 | 0.868 |
| **Atelectasis** | 0.484 | 0.500 | 0.420 | 0.420 |
| **Fibrosis** | 0.477 | 0.595 | 0.590 | 0.590 |
| **No Finding** | 0.498 | 0.758 | 0.618 | 0.618 |

---

### 3️⃣ **Confusion Matrix**
![Confusion Matrix](images/confusion_matrix.png)

🔹 **Description:** The confusion matrix provides insight into the model's predictions for each disease category. 

🔹 **Observations:**
- The model confuses **Consolidation and Effusion**, indicating overlapping features.
- **No Finding** class is sometimes misclassified, suggesting potential label noise.
- Performance for rare classes like **Mass and Calcification** needs improvement.

---

### 4️⃣ **Precision-Recall Curve**
![Precision-Recall Curve](images/PR_curve%20(1).png)

🔹 **Description:** The **Precision-Recall Curve** evaluates the model's classification performance across different confidence thresholds.

🔹 **Key Insights:**
- **Effusion (0.868 mAP)** shows strong performance.
- **Pneumothorax and Calcification** have lower precision, requiring better feature extraction.
- Overall **mAP@0.5 = 0.488**, indicating room for improvement.

---


### **🚀 Next Steps & Future Improvements**
- **Improve Precision:** Reduce false positives by increasing `conf` threshold.
- **Class Imbalance Solutions:** Apply **weighted loss** to handle rare conditions (e.g., Pneumothorax).
- **Hybrid Approach:** Combine YOLO for detection with **ResNet/EfficientNet for classification**.
- **Data Augmentation:** Experiment with **mixup augmentation** to improve generalization.



