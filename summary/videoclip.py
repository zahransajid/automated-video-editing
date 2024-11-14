
import copy
import math
from typing import Any, List, Tuple
import cv2


class VideoClip:
    def __init__(self, path : str, ):
        self.path = path
        videoclip = cv2.VideoCapture(self.path)
        self.fps = videoclip.get(cv2.CAP_PROP_FPS)
        self.original_framecount = int(videoclip.get(cv2.CAP_PROP_FRAME_COUNT))
        self.original_length = self.original_framecount/self.fps
        self.frames = list(range(self.original_framecount))
    
    
    def cut_time_segments(self, segments : Tuple[int, int]):
        for start_time, end_time in segments:
            start_frame = math.floor(start_time*self.fps)
            end_frame = math.floor(end_time*self.fps)
            for i in range(start_frame,end_frame+1):
                self.frames[i] = -1
        self.frames = [frame for frame in self.frames if frame != -1]
    
    def apply_module_results(self, module_results : List[bool]):
        for i in range(len(module_results)):
            if not module_results[i]:
                self.frames[i] = -1
        self.frames = [frame for frame in self.frames if frame != -1]
        self.feature_frames = self.frames.copy()
    
    def map_new_to_old_frame(self, start_frame : int, end_frame : int):
        return self.feature_frames[start_frame:end_frame]
        
    
    
            