import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json

from sklearn.metrics import classification_report, confusion_matrix

# ---- 1. Video-wise Accuracy Table and Plot ----

with open("results/input_videos_accuracy.json", "r") as f:
    data = json.load(f)

video_data = []
for video, info in data[0].items():
    video_data.append({
        "Video": video,
        "Accuracy": info["accuracy_percent"]
    })
overall_accuracy = data[1]

# Convert to DataFrame
df = pd.DataFrame(video_data)
df = df.sort_values("Accuracy", ascending=False, ignore_index=True)
print(df)

# Plotting
plt.figure(figsize=(12, 6))
sns.set(style="whitegrid")
barplot = sns.barplot(data=df, x="Video", y="Accuracy", palette="viridis")

# Add overall accuracy line
plt.axhline(overall_accuracy, color='red', linestyle='--', label=f'Overall Accuracy ({overall_accuracy:.2f}%)')

# Annotate bars
for i, row in df.iterrows():
    barplot.text(i, row["Accuracy"] + 1, f'{row["Accuracy"]:.0f}%', ha='center', va='bottom', fontsize=10)

# Customize plot
plt.title("Per-Video Segment Classification Accuracy", fontsize=16)
plt.ylabel("Accuracy (%)")
plt.ylim(0, 110)
plt.xticks(rotation=30, ha='right')
plt.legend()
plt.tight_layout()

plt.show()

# ---- 2. Confusion Matrix

# Load JSON files
with open("labels/output_labels_test_videos.json") as f:
    pred_data = json.load(f)

with open("labels/input_labels_updated.json") as f:
    manual_data = json.load(f)

# Convert predicted JSON to DataFrame
pred_rows = []
for video, segments in pred_data.items():
    for seg_key, seg_info in segments.items():
        segment_index = int(seg_key.strip("segment_")) if "segment_" in seg_key else int(seg_key)
        pred_rows.append({
            "video": video,
            "segment_index": segment_index,
            "predicted_label": seg_info["label"].lower()
        })
pred_df = pd.DataFrame(pred_rows)

# Convert manual JSON to DataFrame
manual_rows = []
for video, segments in manual_data.items():
    for seg_key, seg_info in segments.items():
        segment_index = int(seg_key)
        manual_rows.append({
            "video": video,
            "segment_index": segment_index,
            "label_manual": seg_info["label"].lower()
        })
manual_df = pd.DataFrame(manual_rows)

# Merge both DataFrames on video and segment_index
merged_df = pd.merge(pred_df, manual_df, on=["video", "segment_index"])

# Evaluate performance
print("🔍 Classification Report:")
print(classification_report(merged_df['label_manual'], merged_df['predicted_label']))

# Confusion matrix
labels = ['converging', 'diverging', 'stable']
cm = confusion_matrix(merged_df['label_manual'], merged_df['predicted_label'], labels=labels)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted Label")
plt.ylabel("Manual Label (Ground Truth)")
plt.title("Confusion Matrix: Predicted vs Manual Labels")
plt.tight_layout()
plt.show()
