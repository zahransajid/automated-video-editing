from abc import ABC, abstractmethod, abstractclassmethod
from dataclasses import dataclass
from enum import Enum
from typing import List
import cv2

@dataclass
class StreamInfo:
    height : int
    width : int
    fps : int

class ParameterType(Enum):
    SliderValue100 = 1
    ToggleOnOffButton = 2

@dataclass
class Parameter:
    parameter_type : ParameterType
    name : str
    description : str
    
    
    

class Module():
    @abstractmethod
    def __init__(self) -> None:
        pass
    @abstractmethod
    def register(self, info : StreamInfo) -> List[Parameter]:
        """Register function called when the Module is imported and set up, it is passed
        parameters of the video stream that it will operate on, and will return named parameters.

        Args:
            info (StreamInfo): Object containing details of the input stream

        Returns:
            List[Parameter]: List of named Parameters that it accepts.
        """
        pass
    @abstractmethod
    def run(self, frame : cv2.Mat) -> None:
        """Run on a single frame passed to the module, save results elsewhere till
        the results function is called elsewhere.

        Args:
            frame (cv2.Mat): Input frame
        """
        pass
    
    @abstractmethod
    def results(self) -> List[float]:
        """Returns a list with a value associated with each frame passed to the module

        Returns:
            List[float]: List of floating point values mapped within range 0-100
        """
        pass