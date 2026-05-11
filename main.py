import json

from detector import Detector
from utils import read_frames, save_csv, load_positions_from_csv
from heuristics import classify_segments_by_slope


def main():
    video_path = "path_to_video.mp4"
    # Read frames
    frames, fps = read_frames(video_path)

    # Detect Bounding Boxes
    detector_model = Detector("yolov8n.pt")
    bboxes = detector_model.get_bboxes(frames)

    # Save Bounding Boxes to CSV file for future use
    save_csv(bboxes, fps, "bboxes_output.csv")

    # Load ground truth labels from JSON file
    with open("ground_truth_label.json", "r") as f:
        gt_labels = json.load(f)

    # Calculate positions w.r.t. time
    positions_by_time = load_positions_from_csv(f"bboxes_output.csv")

    # Classify segments
    classify_segments_by_slope(video_path, gt_labels, positions_by_time, fps,
                               lower_slope_threshold=-0.02,
                               upper_slope_threshold=0.1,
                               output_plot_dir="plots")


if __name__ == '__main__':
    main()
