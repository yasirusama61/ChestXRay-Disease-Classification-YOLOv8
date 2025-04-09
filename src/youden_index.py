import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_curve

# Load predictions and labels
y_true = np.load("/kaggle/working/y_true.npy")
y_probs = np.load("/kaggle/working/y_probs.npy")

n_classes = y_true.shape[1]
youden_thresholds = []

# Compute best thresholds via Youden index
for i in range(n_classes):
    fpr, tpr, thresholds = roc_curve(y_true[:, i], y_probs[:, i])
    j_scores = tpr - fpr
    best_thresh = thresholds[np.argmax(j_scores)]
    youden_thresholds.append(best_thresh)

# Predict using optimal thresholds
y_pred_youden = (y_probs > youden_thresholds).astype(int)

# Report
report = classification_report(y_true, y_pred_youden, target_names=category_to_index.keys(), zero_division=0, output_dict=True)
df = pd.DataFrame(report).transpose()

# Display
print("Youden Thresholds:", youden_thresholds)
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10, 6))
sns.barplot(x=list(category_to_index.keys()), y=youden_thresholds)
plt.xticks(rotation=45)
plt.ylabel("Threshold")
plt.title("📈 Youden's Optimal Thresholds per Class")
plt.tight_layout()
plt.show()

df