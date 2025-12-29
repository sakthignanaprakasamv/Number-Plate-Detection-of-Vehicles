from ultralytics import YOLO

def load_model():
    return YOLO("runs/exp/weights/best.pt")
