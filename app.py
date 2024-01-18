from typing import List
import cv2
import tkinter as tk
from tkinter import ttk, filedialog
from threading import Thread
import os
import json
import numpy as np
import modules
from modules.module import StreamInfo, Parameter
from utils.ffmpegWriter import FFMPEGWriter
from threading import Lock
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
        # self.geometry('300x120')
        self.init_widgets()
        self.pb['value'] = 0
        self.resizable(0,0)
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
        self.pb.grid(column=0, row=1, columnspan=2, padx=10, pady=20)
        # Buttons to start
        self.generate_button = ttk.Button(
            self,
            text='Start Generating function',
            command=self.generate_function
        )
        # Threshold picker
        self.threshold_picker_frame = ttk.Frame(self)
        self.threshold_picker_frame.grid(column=0,row=2)
        self.label_threshold_picker = ttk.Label(
            self.threshold_picker_frame,
            text="Choose a threshold [15]"
        )
        self.label_threshold_picker.pack(side=tk.LEFT,pady=5)
        self.slider_threshold_picker = ttk.Scale(
            self.threshold_picker_frame,
            from_=5,
            to=40,
            command=self.slider_callback
        )
        self.slider_threshold_picker.set(15)
        self.slider_threshold_picker['state'] = 'disabled'
        self.slider_threshold_picker.pack(side=tk.RIGHT,padx=10,pady=5)
        self.generate_button.grid(column=0, row=3, padx=10, pady=5)
        self.export_button = ttk.Button(
            self,
            text='Export video',
            command=self.export_video
        )
        self.export_button.grid(column=0, row=4, padx=10, pady=5)
        # Error box
        self.error_box = tk.Text(
            self,
            height=10,
            width=40
        )
        self.error_box.grid(column=0,row=5,padx=10, pady=10)
    
    def slider_callback(self,*args, **kwargs):
        slider_val = int(self.slider_threshold_picker.get())
        self.label_threshold_picker.config(text=f"Choose a threshold [{slider_val}]")
        self.threshold = slider_val
    
    def pick_file(self):
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
        self.error_box.insert(tk.END,"Starting blood function generation.\n")
        self.compute_lock.acquire()
        input_file = self.fp
        reader = cv2.VideoCapture(input_file)
        total_frames = reader.get(cv2.CAP_PROP_FRAME_COUNT)
        width = reader.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = reader.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = reader.get(cv2.CAP_PROP_FPS)
        stream_info = StreamInfo(height, width, fps)
        frames = 0
        self.modules : List[modules.Module]= [module() for module in modules.module_list]
        for m in self.modules:
            print(type(m))
        self.module_parameters : List[List[Parameter]] = []
        for module in self.modules:
            self.module_parameters.append(module.register(stream_info))
        while True:
            ret, frame = reader.read()
            if not ret:
                break
            for module in self.modules:
                module.run(frame)
            if(frames%100 == 0):
                pb_val = int((frames/total_frames)*100)
                self.pb['value'] = pb_val
            frames += 1
        self.pb['value'] = 100
        self.compute_lock.release()
        self.slider_threshold_picker['state'] = 'normal'
        self.function = True
        self.error_box.insert(tk.END,"Generation complete ready for export.\n")
        
    def write_output(self):
        fp = self.fp
        function = self.function
        self.error_box.insert(tk.END,"Starting video export.\n")
        self.compute_lock.acquire()
        functions = [module.results() for module in self.modules]
        final_function = []
        # naive greater than implementation
        for i in range(len(functions[0])):
            thresh_vals = [val[i] > self.threshold for val in functions]
            thresh_val = (True in thresh_vals)
            final_function.append(thresh_val)
        cap = cv2.VideoCapture(fp)
        # ret, frame_1 = cap.read()
        shape = (1280,720)
        writer = FFMPEGWriter(shape,30,self.export_fp)
        black = np.zeros((shape[0],shape[1],3))
        self.pb["value"] = 0
        current_frame = 0
        max_frame_value = len(function)
        for val in final_function:
            ret,frame = cap.read()
            frame = cv2.resize(frame,shape)
            current_frame += 1
            if(current_frame%100 == 0):
                self.pb["value"] = int((current_frame/max_frame_value)*100)
            if not ret:
                writer.release()
                break
            if(val):
                writer.write(black)
            else:
                writer.write(frame)
        writer.release()
        self.compute_lock.release()
        self.pb['value'] = 100
        self.error_box.insert(tk.END,"Export complete\n")
        pass


if __name__ == '__main__':
    app = App()
