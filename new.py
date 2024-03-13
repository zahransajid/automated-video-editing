from typing import Any, List
import cv2
import tkinter as tk
from tkinter import ttk, filedialog
from threading import Thread
import os
import json
import numpy as np
import modules
from modules.module import StreamInfo, Parameter, ParameterType
from utils.ffmpegWriter import FFMPEGWriter
from utils.tkSliderWidget import Slider
from threading import Lock
from summarizer import *
import copy

import moviepy.editor as mp


USE_THEME = False

class App(tk.Tk):
    def __init__(self, screenName: str | None = None, baseName: str | None = None, className: str = "Tk", useTk: bool = True, sync: bool = False, use: str | None = None) -> None:
        super().__init__(screenName, baseName, className, useTk, sync, use)
        if USE_THEME:
            self.style = ttk.Style(self)
            self.tk.call("source", "breeze.tcl")
            self.style.theme_use("Breeze")
        self.fp = None
        self.export_fp = None
        self.function = False
        self.threshold = 15
        self.compute_lock = Lock()
        self.title("Video Editing System")
        self.grid()
        self.init_widgets()
        self.pb['value'] = 0
        self.resizable(0,0)

        #-----------------summerize----------------#
        self.percentage = 100
        self.keyframes = None
        self.features = {}
        self.videoName = None
        #-----------------summerize----------------#

        #-----------------time based---------------#
        self.current_time = 0
        self.total_duration = 0
        #-----------------time based---------------#

        #-----------------clip---------------------#
        
        self.start = 0
        self.end = 0
        self.video = None
        #-----------------clip---------------------#

        self.mainloop()

    def init_widgets(self):
        # File picker section
        self.file_picker_frame = ttk.Frame(self)
        self.file_picker_frame.grid(column=0,row=0)
        self.label_file_picker = ttk.Label(
            self.file_picker_frame,
            text="Choose input video: "
        )
        self.label_file_picker.pack(side=tk.LEFT)
        self.button_file_picker = ttk.Button(
            self.file_picker_frame,
            text="Browse",
            command=self.pick_file
        )
        self.button_file_picker.pack(side=tk.RIGHT)
        # Progress bar
        self.pb = ttk.Progressbar(
            self,
            orient='horizontal',
            mode='determinate',
            length=280
        )
        self.pb.grid(column=0, row=1, columnspan=10, padx=10, pady=20)

        #---------summerizer button--------------------#
        self.generate_summary_button = ttk.Button(
            self,
            text='Process Video',
            command=self.generate_keyframes
        )
        self.generate_summary_button.grid(column=0, row=7, padx=5, pady=5)
                
        #-------------------summerizer----------------------#

        # Error box
        self.error_box = tk.Text(
            self,
            height=10,
            width=60
        )
        self.error_box.grid(column=0,row=12,padx=10, pady=10)
    

    def get_video_details(self):
        fp = self.fp
        total_time = time_duration(fp)

        self.slider = Slider(
            min_val=0,
            max_val=total_time,
            init_lis=[0, total_time],
            show_value=True,
        )
        self.slider.grid(column=0,row=5,padx=10, pady=10)
        print("Ran this functions")
        return 0
    
    
    def create_parameter_widgets(self):
        def callback(*args,**kwargs):
            for i,v in enumerate(self.current_intvar_store):
                self.parameter_store[self.module_index][i] = v.get()
            print(self.parameter_store[self.module_index])
        parameters = self.module_parameters[self.module_index]
        n = len(parameters)
        for widget in self.parameter_set_frame.winfo_children():
            widget.destroy()
        self.current_intvar_store : List[tk.IntVar] = []
        for i,p in enumerate(parameters):
            if(p.parameter_type == ParameterType.SliderValue100):
                s = tk.IntVar(self)
                self.current_intvar_store.append(s)
                s.set(self.parameter_store[self.module_index][i])
                t = ttk.Scale(
                    self.parameter_set_frame,
                    from_=0,
                    to=100,
                    orient="horizontal",
                    variable=s,
                    command=callback
                )
                t.pack()
                
            elif(p.parameter_type == ParameterType.ToggleOnOffButton):
                s = tk.IntVar(self)
                self.current_intvar_store.append(s)
                s.set(self.parameter_store[self.module_index][i])
                t = ttk.Checkbutton(
                    self.parameter_set_frame,
                    onvalue=1,
                    offvalue=0,
                    variable=s,
                    command=callback
                )
                t.pack()
            
    def update_parameter_frame(self, *args, **kwargs):
        self.module_index = self.module_names.index(self.parameter_choice.get())
        self.create_parameter_widgets()
    def create_module_parameters_box(self):
        if self.fp == "":
            return
        self.parameters_frame = ttk.Frame(self)
        self.parameters_frame.grid(column=0,row=6)
        self.parameters_frame.grid()
        clip = mp.VideoFileClip(self.fp)
        clip = clip.subclip(self.start, self.end)
        fps = clip.fps
        width = clip.size[0] 
        height = clip.size[1]
        stream_info = StreamInfo(height, width, fps)
        self.modules : List[modules.Module]= [module() for module in modules.module_list]
        for m in self.modules:
            print(type(m))
        self.module_parameters : List[List[Parameter]] = []
        for module in self.modules:
            self.module_parameters.append(module.register(stream_info))
        self.parameter_store : List[List[bool | int]] = [[p.default for p in parameters] for parameters in self.module_parameters]
        self.module_index = 0
        self.module_names = [module.name for module in self.modules]
        self.parameter_choice = tk.StringVar(self)
        self.parameter_picker = ttk.OptionMenu(
            self.parameters_frame,
            self.parameter_choice,
            self.module_names[self.module_index],
            *self.module_names,
            command=self.update_parameter_frame
        )
        self.parameter_picker.grid(column=0,row=0)
        self.parameter_set_frame = ttk.Frame(self.parameters_frame)
        self.parameter_set_frame.grid(column=0,row=1)
        self.parameter_set_frame.grid()
        self.create_parameter_widgets()
    
    def pick_file(self):
        if(self.compute_lock.locked()):
            self.error_box.insert(tk.END,"Processing is currently happening please wait for it to end.\n")
            return
        fp = filedialog.askopenfilename(
            initialdir = os.getcwd(),
            title="Select video file",
            filetypes=(("Video files",
                        "*.mp4*"),
                        ("all files",
                        "*.*"))
        )
        if(fp == ""):
            pass
        else:
            self.fp = fp 
            self.label_file_picker.config(text=self.fp.split("/")[-1])
            self.percentage = 100
            self.keyframes = None
            self.features = {}
            self.videoName = None
            self.current_time = 0
            self.total_duration = 0
            self.start = 0
            self.end = 0
            self.video = None
            total_time = time_duration(fp)
            self.slider = Slider(
                self,
                min_val=0,
                max_val=total_time,
                init_lis=[0, total_time],
                show_value=True,
            )
            self.slider.grid(column=0,row=5,padx=15, pady=10)
            self.create_module_parameters_box()


    def pick_export_path(self):
        fp = filedialog.asksaveasfilename(
            initialdir = os.getcwd(),
            title="Export to",
            filetypes=(("Video files",
                        "*.mp4*"),
                        ("all files",
                        "*.*"))
        )
        if(fp == ""):
            pass
        else:
            if(not fp.endswith(".mp4")):
                fp += ".mp4"
            self.export_fp = fp 

    def generate_function(self):
        if(self.compute_lock.locked()):
            self.error_box.insert(tk.END,"Processing is currently happening please wait for it to end.\n")
            return
        if((self.fp is None) or (not os.path.isfile(self.fp))):
            self.error_box.insert(tk.END,"ERROR: Provided file path does not exist!\n")
            return
        try:
            self.t_export = Thread(target=self.generate_functions)
            self.t_export.start()

        except Exception as e:
            self.error_box.insert(tk.END,f"ERROR: {e}\n")
        pass



    def export_video(self):
        if(self.compute_lock.locked()):
            self.error_box.insert(tk.END,"Processing is currently happening please wait for it to end.\n")
            return
        if(self.function == False):
            self.error_box.insert(tk.END,"ERROR: Function has not been generated yet!\n")
            return
        if((self.fp is None) or (not os.path.isfile(self.fp))):
            self.error_box.insert(tk.END,"ERROR: Provided file path to source does not exist!\n")
            return
        self.pick_export_path()
        if(self.export_fp is None):
            self.error_box.insert(tk.END,"ERROR: Please provide a valid export path!\n")
            return
        try:
            t = Thread(target=self.write_output)
            t.start()
        except Exception as e:
            self.error_box.insert(tk.END,f"ERROR: {e}\n")
        pass

    def generate_functions(self):
        self.error_box.insert(tk.END,"Starting Video Cleaning Process.\n")
        input_file = self.fp
        self.start, self.end = self.slider.getValues()
        clip = mp.VideoFileClip(input_file)
        clip = clip.subclip(self.start, self.end)
        total_duration = clip.duration
        fps = clip.fps
        total_frames = total_duration * fps
        subclip = clip.iter_frames()
        width = clip.size[0] 
        height = clip.size[1]
        stream_info = StreamInfo(height, width, fps)
        count = 0
        frames = []
        shape = (1280,720)
        writer = FFMPEGWriter(shape,30)
        while True:
            res = True
            try:
                frame = next(subclip)
                for i,module in enumerate(self.modules):
                    res = res and module.run(frame,self.parameter_store[i])
                if(count%100 == 0):
                    pb_val = int((count/total_frames)*100)
                    self.pb['value'] = pb_val
                if res:
                    writer.write(frame)
                    self.features[count] = get_features(frame)
                    count += 1
            except StopIteration:
                break  
        
        video = writer.release()
        self.video = mp.VideoClip(video)  
        self.pb['value'] = 100
        self.function = True
        self.total_duration = self.video.duration

    def write_output(self):
        video = mp.VideoClip()
        fp = self.fp
        function = self.function
        self.compute_lock.acquire()
        functions = [module.results() for module in self.modules]
        final_function = []
        # naive greater than implementation
        for i in range(len(functions[0])):
            thresh_vals = [val[i] > self.threshold for val in functions]
            thresh_val = (True in thresh_vals)
            final_function.append(thresh_val)
        cap = cv2.VideoCapture(fp)
        shape = (1280,720)
        black = np.zeros((shape[0],shape[1],3))
        self.pb["value"] = 0
        current_frame = 0
        max_frame_value = len(functions)
        for val in final_function:
            ret,frame = cap.read()
            frame = cv2.resize(frame,shape)
            current_frame += 1
            if(current_frame%100 == 0):
                self.pb["value"] = int((current_frame/max_frame_value)*100)
            if not ret:
                break
            if(val):
                video.write_frame(frame)
        self.compute_lock.release()
        self.pb['value'] = 100
        self.error_box.insert(tk.END,"Video Cleaning Done!!.\n")
        self.video = video
    
    #-------------------summerizer----------------------#


    def generate_keyframes(self):
        if(self.compute_lock.locked()):
            self.error_box.insert(tk.END,"Processing is currently happening please wait for it to end.\n")
            return
        if((self.fp is None) or (not os.path.isfile(self.fp))):
            self.error_box.insert(tk.END,"ERROR: Provided file path to source does not exist!\n")
            return
        try:
            t = Thread(target=self.keyframe_generator)
            t.start()
        except Exception as e:
            self.error_box.insert(tk.END,f"ERROR: {e}\n")
        pass

    def keyframe_generator(self):
        fp = self.fp
        self.compute_lock.acquire()
        self.videoName = fp.split("/")[-1][:-4]
        self.generate_functions()
        self.error_box.insert(tk.END,"Starting Summerization Task for "+self.videoName+".mp4\n")
        self.pb['value'] = 50
        self.error_box.insert(tk.END,"Generating Key Frames...\n")
        self.keyframes = generate_keyframes(self.features)
        self.pb['value'] = 100
        total_frames = len(list(self.features.values()))
        video_details = [self.videoName, total_frames, self.keyframes]
        save_details("./video_details/"+self.videoName+".csv", video_details)
        self.error_box.insert(tk.END,"Key Frames Generated Ready for Summarization !!!\n")

        #------------------percentage based-----------------#
        self.summerize_button = ttk.Button(
            self,
            text='Summerize',
            command=self.summerizer
        )
        self.summerize_button.grid(column=0, row=9, padx=10, pady=5)
        self.percentage_picker_frame = ttk.Frame(self)
        self.percentage_picker_frame.grid(column=0,row=8)
        self.label_percentage_picker = ttk.Label(
            self.percentage_picker_frame,
            text="Choose a Percentage to summerize[100]"
        )
        self.label_percentage_picker.pack(side=tk.LEFT,pady=5)
        self.slider_percentage_picker = ttk.Scale(
            self.percentage_picker_frame,
            from_=10,
            to=100,
            command=self.percentage_slider_callback
        )
        self.slider_percentage_picker.set(100)
        self.slider_percentage_picker['state'] = 'normal'
        self.slider_percentage_picker.pack(side=tk.RIGHT,padx=10,pady=5)
        #------------------percentage based-----------------#

        #-------------------time based----------------------#

        self.time_summary_button = ttk.Button(
            self,
            text='Timed Summerization',
            command=self.time_summary
        )
        self.time_summary_button.grid(column=0, row=11, padx=10, pady=5)


        self.time_picker_frame = ttk.Frame(self)
        self.time_picker_frame.grid(column=0,row=10)
        self.label_time_picker = ttk.Label(
            self.time_picker_frame,
            text="Choose a Time to summerize[00:00:00]"
        )
        self.label_time_picker.pack(side=tk.LEFT,pady=5)
        self.slider_time_picker = ttk.Scale(
            self.time_picker_frame,
            from_=10,
            to=self.total_duration,
            command=self.time_slider_callback
        )
        self.slider_time_picker['state'] = 'normal'
        self.slider_time_picker.pack(side=tk.RIGHT,padx=10,pady=5)

        #-------------------time based----------------------#



        self.compute_lock.release()

    def percentage_slider_callback(self,*args, **kwargs):
        slider_val = int(self.slider_percentage_picker.get())
        self.label_percentage_picker.config(text=f"Choose a Percentage to summerize[{slider_val}]")
        self.percentage = slider_val

    def summerizer(self):
        if(self.compute_lock.locked()):
            self.error_box.insert(tk.END,"Processing is currently happening please wait for it to end.\n")
            return
        if((self.fp is None) or (not os.path.isfile(self.fp))):
            self.error_box.insert(tk.END,"ERROR: Provided file path to source does not exist!\n")
            return
        try:
            t = Thread(target=self.summerize)
            t.start()
        except Exception as e:
            self.error_box.insert(tk.END,f"ERROR: {e}\n")
        pass

    def summerize(self):
        if self.keyframes is None:
            self.error_box.insert(tk.END,"First Generate Keyframes!!!\n")
            return
        fp = self.fp
        videoName = self.videoName

        self.compute_lock.acquire()
        keyframes = self.keyframes
        event_boundary_threshold = 0.99
        percentage = self.percentage
        self.error_box.insert(tk.END,"Processing video...\n")
        event_boundaries,ebt = boundary_determination(self.features, keyframes, event_boundary_threshold, percentage)
        self.pick_export_path()
        self.error_box.insert(tk.END,"Generating Summary...\n\n")
        output_summary(self.video, event_boundaries,self.export_fp)
        self.error_box.insert(tk.END,str(percentage)+"% Summary saved into: "+self.export_fp+"\n\n")
        self.compute_lock.release()
    #-------------------summerizer----------------------#

    def time_summary(self):
        if(self.compute_lock.locked()):
            self.error_box.insert(tk.END,"Processing is currently happening please wait for it to end.\n")
            return
        if((self.fp is None) or (not os.path.isfile(self.fp))):
            self.error_box.insert(tk.END,"ERROR: Provided file path to source does not exist!\n")
            return
        try:
            t = Thread(target=self.time_summerizer)
            t.start()
        except Exception as e:
            self.error_box.insert(tk.END,f"ERROR: {e}\n")
        pass

    def time_summerizer(self):
        if self.keyframes is None:
            self.error_box.insert(tk.END,"First Generate Keyframes!!!\n")
            return
        fp = self.fp
        videoName = self.videoName
        self.compute_lock.acquire()
        keyframes = self.keyframes
        seconds = self.current_time
        self.error_box.insert(tk.END,"Processing video...\n")
        event_boundaries = time_boundary_determination(self.features, keyframes, seconds, 30)
        hours, minutes, seconds  = self.get_time(seconds)
        self.error_box.insert(tk.END,"Generating Summary...\n\n")
        self.pick_export_path()
        output_summary(self.video, event_boundaries,self.export_fp)
        self.error_box.insert(tk.END,f"{hours:02d}:{minutes:02d}:{seconds:02d} Summary saved into: {self.export_fp}\n\n")
        self.compute_lock.release()

    def get_time(self, seconds):
        hrs = seconds // 3600
        mins = (seconds % 3600) // 60
        secs = seconds % 60
        return hrs, mins, secs
    
    def time_slider_callback(self,*args, **kwargs):
        slider_val = int(self.slider_time_picker.get())
        position = int(slider_val)
        hours, minutes, seconds  = self.get_time(position)
        self.label_time_picker.config(text=f"Choose a Time to summerize[{hours:02d}:{minutes:02d}:{seconds:02d}]")
        self.current_time = slider_val


if __name__ == '__main__':
    app = App()
