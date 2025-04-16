# scripts/plot_class_distribution.py
import json
import matplotlib.pyplot as plt
from collections import Counter

with open("merged_annotations.json", "r") as f:
    data = json.load(f)

label_counts = Counter()
for entry in data:
    for label in entry["syms"]:
        label_counts[label] += 1

plt.figure(figsize=(10, 5))
plt.bar(label_counts.keys(), label_counts.values())
plt.xticks(rotation=45)
plt.title("Class Distribution in Merged Dataset")
plt.tight_layout()
plt.savefig("images/class_distribution.png")
plt.show()
