from typing import Any, List

from cv2 import Mat
from modules.module import Parameter, StreamInfo, ParameterType
from .module import Module
import numpy as np
import cv2
import time as t


class FilterFunction():
    def __init__(self, f : np.ndarray) -> None:
        self.function = f
        self.smooth_function = self.smooth(self.function)
        self.fps = 30
    def smooth(self, function : np.ndarray):
        fft = np.fft.rfft(function)
        mask = np.ones(fft.shape)
        # Update criteria for fft selection
        for i in range(len(mask)//1000,len(mask)):
            mask[i] = 0
        fft *= mask
        return np.fft.irfft(fft)

    def get_function(self, threshold_value : int, smooth = True) -> List[int]:
        if(smooth):
            return [False if val > threshold_value else True for val in self.smooth_function]
        else:
            return [False if val > threshold_value else True for val in self.function]
    def get_percentage(self, threshold_value : int):
        percentage = np.average(self.get_function(threshold_value))/(self.fps)
        remaining_length = len(self.function)//self.fps
        return f"You will trim off {percentage//60}m and {percentage%60} seconds.\
                This will result in a clip that is {(remaining_length - percentage)//60}m long."


class EditableFilter():
    def __init__(self,*args, **kwargs):    
        self.lower = np.array([0,0,0],dtype=np.int32)
        self.upper = np.array([360,255,255],dtype=np.int32)
        if('lower' in kwargs.keys()):
            self.lower = np.array(kwargs['lower'],dtype=np.int32)
        if('upper' in kwargs.keys()):
            self.upper = np.array(kwargs['upper'],dtype=np.int32)
        self.JUMP_AMOUNT = 5
        self.KEY_MAP = {
            'q' : [1,0,0],
            'a' : [-1,0,0],
            'w' : [0,1,0],
            's' : [0,-1,0],                              
            'e' : [0,0,1],
            'd' : [0,0,-1],
            'r' : [1,0,0],
            'f' : [-1,0,0],
            't' : [0,1,0],
            'g' : [0,-1,0],
            'y' : [0,0,1],
            'h' : [0,0,-1],
        }
        for k in self.KEY_MAP.keys():
            self.KEY_MAP[k] = np.array(self.KEY_MAP[k],dtype=np.int32)*self.JUMP_AMOUNT
            
    def get_filter(self):
        return (self.lower,self.upper)
    def on_key(self, key_code):
        if(key_code == -1): return
        upper_keys = list('qweasd')
        lower_keys = list('rtyfgh')
        key = chr(key_code).lower()
        if(key in upper_keys):
            self.upper += self.KEY_MAP[key]
        elif(key in lower_keys):
            self.lower += self.KEY_MAP[key]
        else:
            pass

def hsv_filter(img: cv2.Mat, filter_ranges : List):
    start = t.time()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Define bounds for colour based filtering of images
    # The filtering used is to fliter by hue with rough match of saturation
    # Value is disregarded due to lighting conditions
    result_mask = None
    for ranges in filter_ranges:
        lower,upper = ranges
        if result_mask is None:
            result_mask : cv2.Mat = cv2.inRange(hsv, lower, upper)
        else:
            mask = cv2.inRange(hsv, lower, upper)
            cv2.bitwise_or(result_mask,mask, result_mask)
    return (result_mask,img)


class BloodPercentageModule(Module):
    def __init__(self) -> None:
        self.name = "Blood Percentage"
        self.blood_filter = EditableFilter(lower=[160 ,150,60], upper=[360,235,110])
        self.base_function = []
    def register(self, info: StreamInfo) -> List[Parameter]:
        return [Parameter(ParameterType.SliderValue100,"Blood Percentage Threshold","Sets the threshold for how much blood should be there",12)]
    def run(self, frame: Mat, parameters : List[int]) -> None:
        thresh,img = hsv_filter(frame, [self.blood_filter.get_filter(),])
        ratio_black = cv2.countNonZero(thresh)/(thresh.shape[0] * thresh.shape[1])
        val = float(np.round(ratio_black*100, 2))
        self.base_function.append(val)

    def results(self, parameters : List[int]) -> List[float]:
        return FilterFunction(self.base_function).get_function(parameters[0],smooth=False)