 
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
yolo_chestxray/ 
│── data/ # Dataset files (DO NOT upload real patient data) 
│── models/ # Trained YOLOv8 models 
│── results/ # Evaluation results 
│── src/ # Python scripts (training, preprocessing, etc.) 
│── README.md # Project documentation 
│── .gitignore # Ignore large files 
│── requirements.txt # Dependencies
