from typing import List
import numpy as np
import cv2
from cv2 import Mat
from modules.module import Parameter, StreamInfo
from .module import Module


def canny(frame, threshold_low=50, threshold_high=150):
    edges = cv2.Canny(frame, threshold_low, threshold_high)
    edge_percentage = (np.count_nonzero(edges) / (edges.shape[0] * edges.shape[1])) * 100
    blur_threshold = 3
    return edge_percentage < blur_threshold

class BlurDetection(Module):
    def __init__(self) -> None:
        self.result = []
        self.name = "Blur Detector"
    def register(self, info: StreamInfo) -> List[Parameter]:
        return []
    def run(self, frame: Mat, parameters : List[int]) -> None:
        output = canny(frame)
        if output:
            return True
            self.result.append(1)
        else:
            return False
            self.result.append(0)

    def results(self, parameters : List[int]) -> List[float]:
        return self.result
    


