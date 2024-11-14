import json
import time
from typing import Any, List
from modules import Module
from moviepy.editor import VideoFileClip
from .summarizer import Summarizer
import torch
import torch.nn as nn
from torchvision import models
import numpy as np
from scipy.spatial.distance import cdist
import cv2
import moviepy.editor as mp
from sklearn.mixture import GaussianMixture
from sklearn.metrics import davies_bouldin_score
import csv


class Cleaner():
    def __init__(self, module_list : List[Module], debug = False) -> None:
        self.module_list = module_list
        self.debug = debug
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.resnet18 = models.resnet18(weights=None)
        self.resnet18 = self.resnet18.to(self.device)  # Move model to device (GPU or CPU)
        self.resnet18.eval()  # Set model to evaluation mode

        modules18 = list(self.resnet18.children())[:-1]
        self.model = nn.Sequential(*modules18)


    def generate_functions(self, clip : VideoFileClip, parameters : List[Any], generate_features = False):
        subclip = clip.iter_frames()
        self.features = []
        results = []
        time_per_module = [0 for i in range(len(self.module_list))]
        runs_per_module = [0 for i in range(len(self.module_list))]
        if not self.debug:
            for frame_number, frame in enumerate(subclip):
                flag = True
                for i, module in enumerate(self.module_list):
                    t = time.perf_counter()
                    intermediate_result = module.run(frame,parameters[i])
                    time_per_module[i] += time.perf_counter() - t
                    runs_per_module[i] += 1
                    if(intermediate_result == False):
                        results.append(False)
                        flag = False
                        break
                
                if flag:
                    results.append(True)
                    if(generate_features):
                        self.features.append(self.get_features(frame))
        else:
            intermediate_result = []
            for frame_number, frame in enumerate(subclip):
                for i, module in enumerate(self.module_list):
                    intermediate_result.append(module.run(frame,parameters[i]))
                
                results.append(intermediate_result)
        self.results = results.copy()
        with open("video_details/times","w") as f:
            json.dump([
                time_per_module,
                runs_per_module,
                [module.name for module in self.module_list],
                
            ],
                      f)
        return results

    def get_all_features(self):
        return self.features
        
    
    def get_features(self, frame):
        img = frame
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224,224))
        img = torch.Tensor(img)
        img = img.permute(2, 0, 1)
        img = img.unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model(img).view(512)
        if torch.cuda.is_available():
            features = features.cpu().detach().numpy()
        else:
            features = features.detach().numpy()
        return features