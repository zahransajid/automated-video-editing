from typing import List

from cv2 import Mat
from modules.module import Parameter, StreamInfo
from .module import Module
from ultralytics import YOLO


class Instrument(Module):
    def __init__(self) -> None:
        self.name = "Instrument Detector"
        self.path = './model/best.pt'
        self.model = YOLO(self.path,verbose=False)
        self.detections = []
    def register(self, info: StreamInfo) -> List[Parameter]:
        return []
    def run(self, frame: Mat, parameters : List[int]) -> None:
        res = self.model(source = frame, save=False, save_txt=False, imgsz=640, conf = 0.55, device=0, verbose=False)
        if len(res[0]) > 0:
            # return True
            self.detections.append(True)
        else:
            # return False
            self.detections.append(False)

    def results(self, parameters : List[int]) -> List[float]:
        return self.detections
    


