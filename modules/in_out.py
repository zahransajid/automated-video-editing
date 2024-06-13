from typing import List

from cv2 import Mat
from modules.module import Parameter, StreamInfo
from .module import Module
import tensorflow as tf
import numpy as np
import cv2
from io import BytesIO
from PIL import Image
import os


class InOutDetector(Module):
    def __init__(self) -> None:
        self.name = "InSideOutside Detector"
        self.result = []
    
    def register(self, info: StreamInfo) -> List[Parameter]:
        self.model = tf.keras.models.load_model('./model/inside_outside.h5')
        return []
    
    def run(self, frame: Mat, parameters: List[int]) -> None:
        tf.keras.config.disable_interactive_logging()
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = img.resize((150,150), Image.NEAREST)
        img = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img, 0)
        predictions = self.model.predict(img_array)
        self.result.append(predictions[0][0] == 0.0)
    
    def results(self, parameters: List[int]) -> List[bool]:
        return self.result
    
    