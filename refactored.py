from typing import List
from summary.cleaning import Cleaner
import moviepy.editor as mp
from modules import module_list
from modules.module import StreamInfo, Module

TEST_PATH = r"E:\ResearchVideos\First_Batch\A0003\V1221_20361203_081219_7.mp4"


if __name__ == '__main__':
    clip = mp.VideoFileClip(TEST_PATH)
    clip = clip.subclip(8*60, 10*60 + 6)
    
    width = clip.size[0] 
    height = clip.size[1]
    fps = clip.fps
    stream_info = StreamInfo(height, width, fps)
    
    module_list : List[Module] = [module() for module in module_list]
    module_list = sorted(module_list, key= lambda m : m.priority)
    for m in module_list:
        m.register(stream_info)
    print(module_list)
    cleaner = Cleaner(module_list)
    
    parameters = [[17,],[],[]]
    
    results = cleaner.generate_functions(clip,parameters)
    
    print(results, len(results))
    
    