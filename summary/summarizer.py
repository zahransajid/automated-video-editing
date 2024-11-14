import json
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


class Summarizer():
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.resnet18 = models.resnet18(weights=None)
        self.resnet18 = self.resnet18.to(self.device)  # Move model to device (GPU or CPU)
        self.resnet18.eval()  # Set model to evaluation mode

        modules18 = list(self.resnet18.children())[:-1]
        self.model = nn.Sequential(*modules18)

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
        
    def eratosthenis(self, img_features, n):
        prime_numbers = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        segments = [[] for _ in range(n)]
        for fn, fv in img_features.items():
            frame_number = fn
            assigned = False
            for i in range(n):
                if frame_number % prime_numbers[i] == 0:
                    segments[i].append(fv)
                    assigned = True
                    break
            if not assigned:
                segments[-1].append(fv)
        return segments
        
    def optimal_k(self, feature_vectors):
        dbi_scores = []
        k_values = range(3, min(len(feature_vectors), 11))
        for k in k_values:
            gmm = GaussianMixture(n_components=k, random_state=0)
            gmm.fit(feature_vectors)
            dbi_score = davies_bouldin_score(
                feature_vectors, gmm.predict(feature_vectors))
            dbi_scores.append(dbi_score)
        dbi_scores = np.array(dbi_scores)
        index = np.argmin(dbi_scores)
        optimal_k = k_values[index]
        return optimal_k

    def cluster(self, data):
        key_frames = []
        k = self.optimal_k(data)
        gmm = GaussianMixture(n_components=k)
        gmm.fit(data)
        centers = gmm.means_
        for i in range(len(centers)):
            key_array = centers[i]
            distances = cdist(data, np.expand_dims(
                key_array, axis=0), metric="euclidean")
            closest_index = np.argmin(distances)
            closest_array = data[closest_index]
            key_frames.append(closest_array)
        return key_frames

    def generate_keyframes(self, img_features):
        sets = self.eratosthenis(img_features, 5)
        key_frames = []
        final_kf = []
        for idx, set in enumerate(sets):
            var = self.cluster(set)
            key_frames.extend(var)
        for i in key_frames:
            final_kf.append(self.frame_mapper(img_features, i))
        final_kf.sort()
        return final_kf


    def frame_mapper(self, dictionary, value):
        for key, val in dictionary.items():
            if np.array_equal(val, value):
                return key

    def boundary_determination(self, feature_vectors_dict, keyframes, initial_event_boundary_threshold, percentage):
        total_frames = len(feature_vectors_dict)
        max_total_length = int(total_frames * (percentage/100))
        merged_events = [[keyframe, keyframe] for keyframe in keyframes]
        total_length = len(keyframes)
        event_boundary_threshold = initial_event_boundary_threshold
        while total_length < max_total_length:
            next_events = [[max(start-1, 0), min(end + 1, total_frames)] for start, end in merged_events]
            next_total_length = sum(end_frame - start_frame + 1 for start_frame, end_frame in next_events)
            # if next_total_length > max_total_length:
            #     event_boundary_threshold *= 0.9
            # else:
            #     event_boundary_threshold *= 1.1
            merged_events = next_events
            total_length = next_total_length
        for i in range(1,len(merged_events)):
            if merged_events[i-1][1]>=merged_events[i][0]:
                merged_events[i][0] = merged_events[i-1][1]+1
        return merged_events,event_boundary_threshold



    def output_summary(self, frame_ranges, frame_mapper, video_start_frame):
        # video_clip = video_clip.set_fps(fps)
        # fps = video_clip.fps
        # clips = []
        complete_frame_ranges = []
        for start_frame, end_frame in frame_ranges:
            frames = frame_mapper(start_frame, end_frame)
            complete_frame_ranges.extend(frames)
        
        
        segments = []
        n_frames = len(complete_frame_ranges)
        i = 0
        for i in range(n_frames):
            if i == 0:
                segment_start = complete_frame_ranges[i]
                segment_end = complete_frame_ranges[i]
            else:
                if(complete_frame_ranges[i] == segment_end + 1) and i!= n_frames - 1:
                    segment_end += 1
                else:
                    segments.append((segment_start+video_start_frame,segment_end+video_start_frame))
                    segment_end = segment_start = complete_frame_ranges[i]

        return segments
                              
        
        return complete_frame_ranges

    def save_details(self, file_path, video_details):
        with open(file_path, mode="w+", newline="") as csvfile:
            writer = csv.writer(csvfile)
            header = [
                "Video Name",
                "Total Frames",
                "Keyframes",
            ]
            writer.writerow(header)
            writer.writerow(video_details)

    def time_duration(self, video):
        clip = mp.VideoFileClip(video)
        return int(clip.duration)


    def time_boundary_determination(self, feature_vectors_dict, keyframes, seconds, fps):
        total_frames = len(feature_vectors_dict)
        max_total_length = seconds*fps
        merged_events = [[keyframe, keyframe] for keyframe in keyframes]
        total_length = len(keyframes)
        while total_length < max_total_length:
            next_events = [[max(start-1, 0), min(end + 1, total_frames)] for start, end in merged_events]
            i = 1
            while i<len(next_events):
                if next_events[i-1][1]>=next_events[i][0]:
                    next_events[i-1][1] = next_events[i][1]
                    next_events.pop(i)
                else:
                    i+=1
            next_total_length = sum(end_frame - start_frame + 1 for start_frame, end_frame in next_events)
            merged_events = next_events
            total_length = next_total_length
        for i in range(1,len(merged_events)):
            if merged_events[i-1][1]>=merged_events[i][0]:
                merged_events[i][0] = merged_events[i-1][1]+1
        return merged_events


# General flow
# clean video -> generate features (dict of frame number to feature)
# from features determine which ones are keyframes and using previous dict
# Reverse and find their frame numbers
# Pass to boundary determination to find merged events
# Merged events are in format start, end frame
# These are converted to subclips and exported