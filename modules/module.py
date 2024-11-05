from abc import ABC, abstractmethod, abstractclassmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Union
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
    default : Union[bool, int]
    
    
    
# Name is the name of the module, priority is higher if it runs faster
# 1 for CV and 2 for DL

class Module():
    @abstractmethod
    def __init__(self) -> None:
        self.name : str
        self.priority : int
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
    def run(self, frame : cv2.Mat, parameters : List[int]) -> bool:
        """Run on a single frame passed to the module, save results elsewhere till
        the results function is called elsewhere.

        Args:
            frame (cv2.Mat): Single frame passed as input
            parameters (List[int]): Parameter values requested when the module was registered
        """
        pass
    
    def __str__(self) -> str:
        return self.name