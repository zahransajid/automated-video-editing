from .blood_percentage import BloodPercentageModule as _blood_percentage
from .instrument import Instrument as _instrument
from .in_out import InOutDetector as _inout
from .blur_detection import BlurDetection as _blur_detection
from .module import Module
from typing import List

# module_list : List[Module] = [_blood_percentage,_instrument, _blur_detection,] # 

# module_list : List[Module] = [_instrument]

module_list : List[Module] = [_inout]