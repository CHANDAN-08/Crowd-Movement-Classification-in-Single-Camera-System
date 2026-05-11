import csv
import cv2
from collections import defaultdict


def read_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    return frames, fps


def save_csv(all_bboxes, fps, path=None):
    data = []
    for frame, bboxes in all_bboxes:
        for (x, y, w, h) in bboxes:
            data.append((frame, round(frame/fps, 3), x, y, w, h))

    df = pd.DataFrame(data, columns=["frame", "time", "x", "y", "w", "h"])
    df.to_csv(path, index=False)


def load_positions_from_csv(csv_path):
    positions_by_time = defaultdict(list)  # {frame_id: [(x_center, y_center), ...]}

    with open(csv_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            try:
                frame_id = int(row[0])
                x = float(row[2])
                y = float(row[3])
                w = float(row[4])
                h = float(row[5])
                x_center = x + w / 2
                y_center = y + h / 2
                positions_by_time[frame_id].append((x_center, y_center))
            except ValueError as e:
                print(e)
                continue

    return dict(positions_by_time)
