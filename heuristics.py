import os
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from collections import defaultdict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report


def compute_centroid(points):
    xs, ys = zip(*points)
    return np.mean(xs), np.mean(ys)


def compute_dispersion(points):
    if not points:
        return 0.0
    cx, cy = compute_centroid(points)
    return np.mean([np.sqrt((x - cx) ** 2 + (y - cy) ** 2) for x, y in points])


def classify_segments_by_slope(video, gt_labels, positions_by_time, fps,
                               lower_slope_threshold, upper_slope_threshold, segment_duration=10,
                               output_plot_dir='plots'):
    os.makedirs(output_plot_dir, exist_ok=True)
    os.makedirs(f"{output_plot_dir}/{video}", exist_ok=True)

    frame_numbers = sorted(positions_by_time.keys())
    total_frames = len(frame_numbers)
    frames_per_segment = int(fps * segment_duration)
    num_segments = total_frames // frames_per_segment + int(total_frames % frames_per_segment != 0)

    # Compute dispersion for all frames
    dispersion_time_series = []
    for frame in frame_numbers:
        points = positions_by_time[frame]
        dispersion = compute_dispersion(points)
        dispersion_time_series.append((frame, dispersion))

    # Segment-wise classification
    segment_labels = {}
    csv_data = []
    for i in range(num_segments):
        start_idx = i * frames_per_segment
        end_idx = min((i + 1) * frames_per_segment, total_frames)
        segment_data = dispersion_time_series[start_idx:end_idx]
        if not segment_data:
            continue

        gt_label = gt_labels[str(i)]["label"]
        x = list(range(len(segment_data)))  # Relative frame index within segment
        y = [disp for _, disp in segment_data]

        # Linear regression
        slope, intercept, _, _, _ = linregress(x, y)

        # Heuristic classification using slope
        if slope < lower_slope_threshold:
            label = "Converging"
        elif slope > upper_slope_threshold:
            label = "Diverging"
        else:
            label = "Stable"

        # Dispersion stats
        avg_dispersion = round(np.mean(y), 4)
        std_dispersion = round(np.std(y), 4)
        dispersion_change = round(y[-1] - y[0], 4)
        segment_length = len(segment_data)

        # Save plot
        segment_frames = [f for f, _ in segment_data]
        plt.figure(figsize=(8, 4))
        plt.plot(x, y, 'bo-', label='Dispersion')
        plt.plot(x, [slope * xi + intercept for xi in x], 'r--', label=f'Slope={slope:.3f}')
        plt.title(f"Segment {i}: {label}")
        plt.xlabel("Frame Index (relative)")
        plt.ylabel("Dispersion")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_plot_dir, video, f"segment_{i}.png"))
        plt.close()

        # Save to dict
        segment_labels[f"segment_{i}"] = {
            "start_frame": segment_frames[0],
            "end_frame": segment_frames[-1],
            "slope": round(slope, 4),
            "average_dispersion": avg_dispersion,
            "dispersion_std_dev": std_dispersion,
            "dispersion_change": dispersion_change,
            "segment_length": segment_length,
            "label": label,
            "gt_label": gt_label
        }

        # Save to CSV
        csv_data.append([
            video,
            f"segment_{i}",
            segment_frames[0],
            segment_frames[-1],
            round(slope, 4),
            avg_dispersion,
            std_dispersion,
            dispersion_change,
            segment_length,
            label,
            gt_label
        ])

    # Save JSON
    with open(f"{output_plot_dir}/{video}/output_labels.json", "w") as f:
        json.dump(segment_labels, f, indent=2)

    # Save CSV
    csv_file_path = f"{output_plot_dir}/{video}/{video}_segment_data.csv"
    print(csv_file_path)
    with open(csv_file_path, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "video", "segment_id", "start_frame", "end_frame", "slope_dispersion",
            "avg_dispersion", "dispersion_std_dev", "dispersion_change",
            "segment_length_frames", "pred_label", "gt_label"
        ])
        writer.writerows(csv_data)

    return segment_labels
