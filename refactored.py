from typing import List
from summary.cleaning import Cleaner
from summary.summarizer import Summarizer
from summary.videoclip import VideoClip
import moviepy.editor as mp
from modules import module_list
from modules.module import StreamInfo, Module
from cProfile import Profile
from pstats import Stats,SortKey

TEST_PATH = r"E:\ResearchVideos\First_Batch\A0003\V1221_20361203_081219_7.mp4"


if __name__ == '__main__':
    clip = mp.VideoFileClip(TEST_PATH)
    clip = clip.subclip(8*60, 8*60 + 10)
    
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

    results = cleaner.generate_functions(clip,parameters,generate_features=True)
    vc = VideoClip(TEST_PATH)
    vc.apply_module_results(results)
    summarizer = Summarizer()
    features = {i : feature for i, feature in enumerate(cleaner.get_all_features())}
    keyframe_numbers = summarizer.generate_keyframes(features)
    event_boundaries,ebt = summarizer.boundary_determination(features, keyframe_numbers, 0.99, 70)
    segments = summarizer.output_summary(event_boundaries,vc.map_new_to_old_frame, 0)
    time_segments = [
        (start_frame/fps, end_frame/fps) for start_frame, end_frame in segments
    ]
    clips = []
    for start_time, end_time in time_segments:
        sclip = clip.subclip(start_time,end_time)
        clips.append(sclip)
    final_clip = mp.concatenate_videoclips(clips)
    final_clip.write_videofile("video_summaries/output.mp4", codec="libx264", audio_codec="aac")
    
    
    
    
    