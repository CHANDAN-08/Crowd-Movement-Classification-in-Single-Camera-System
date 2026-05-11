import os
import pickle
import numpy as np
from ultralytics import YOLO


class Detector:
    def __init__(self, model_path="yolov8x.pt"):
        self.model = YOLO(model_path)

    def detect_frames(self, frames):
        detections = []
        for frame in frames:
            d = self.model.predict(frame, device="cuda:0", conf=0.2)
            detections += d
        return detections

    def get_bboxes(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                detections = pickle.load(f)
        else:
            detections = self.detect_frames(frames)
            if stub_path is not None:
                with open(stub_path, "wb") as f:
                    pickle.dump(detections, f)

        bboxes = []
        for frame_num, detection in enumerate(detections):
            boxes = []
            for bbox in detection.boxes:
                cls = int(bbox.cls[0])
                xywh = np.array(bbox.xywh[0].cpu(), dtype=np.int32)
                if cls != 0:
                    continue
                boxes.append((xywh[0], xywh[1], xywh[2], xywh[3]))
            bboxes.append((frame_num, boxes))

        return bboxes
