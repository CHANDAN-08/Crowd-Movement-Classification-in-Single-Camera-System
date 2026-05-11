import json
import os

from sklearn.metrics import classification_report

with open("labels/input_labels_80.json", "r") as f:
    ground_truth = json.load(f)

dirs = os.listdir("plots_new_threshold/input_videos")
predicted_labels = {}
for vid in dirs:
    with open(f"plots_new_threshold/input_videos/{vid}/output_labels.json", "r") as f:
        predicted_labels[vid] = json.load(f)


def calculate_accuracy(actual, predicted):
    total_segments = 0
    correct_segments = 0

    per_video_results = {}

    for video, segments in actual.items():
        video_total = len(segments)
        video_correct = 0

        try:
            predicted_segments = predicted[video]
        except KeyError as e:
            print(e)
            continue

        for seg_id, gt_data in segments.items():
            gt_label = gt_data["label"].lower()
            pred_label = predicted_segments[f"segment_{seg_id}"]["label"].lower()

            if gt_label == pred_label:
                video_correct += 1

        per_video_results[video] = {
            "total_segments": video_total,
            "correct_segments": video_correct,
            "accuracy_percent": (video_correct / video_total) * 100 if video_total > 0 else 0
        }

        total_segments += video_total
        correct_segments += video_correct

    overall_accuracy = (correct_segments / total_segments) * 100 if total_segments > 0 else 0

    # Prepare flat lists for classification report
    y_true = []
    y_pred = []

    for video, segments in ground_truth.items():
        if video not in predicted_labels:
            continue

        for seg_id, gt_data in segments.items():
            gt_label = gt_data["label"].lower()

            try:
                pred_label = predicted_labels[video][f"segment_{seg_id}"]["label"].lower()
            except KeyError:
                pred_label = "unknown"  # or handle missing segments differently

            y_true.append(gt_label)
            y_pred.append(pred_label)

    # Print classification report
    print(classification_report(y_true, y_pred, digits=2))

    return per_video_results, overall_accuracy


# Run the function
per_video_results, overall_accuracy = calculate_accuracy(ground_truth, predicted_labels)

# Print detailed results
for video, results in per_video_results.items():
    print(f"Video: {video}")
    print(f"  Total segments: {results['total_segments']}")
    print(f"  Correctly predicted: {results['correct_segments']}")
    print(f"  Accuracy: {results['accuracy_percent']:.2f}%")
    print()

print(f"Overall accuracy across all videos: {overall_accuracy:.2f}%")