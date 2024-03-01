import torch
import torch.nn as nn
from torchvision import models
import numpy as np
from scipy.spatial.distance import cdist
import cv2
import moviepy.editor as mp
from sklearn.mixture import GaussianMixture
from sklearn.metrics import davies_bouldin_score
from tqdm import tqdm
import os
import csv

# Process finished --- 11015.14189863205 seconds ---


# import time
# start_time = time.time()

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resnet18 = models.resnet18(weights=None)
resnet18 = resnet18.to(device)  # Move model to device (GPU or CPU)
resnet18.eval()  # Set model to evaluation mode

modules18 = list(resnet18.children())[:-1]
model = nn.Sequential(*modules18)
flatten = nn.Flatten()

def get_features(frame):
    img = frame
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224,224))
    img = torch.Tensor(img)
    img = img.permute(2, 0, 1)
    img = img.unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(img).view(512)
    if torch.cuda.is_available():
        features = features.cpu().detach().numpy()
    else:
        features = features.detach().numpy()
    return features
    
def extract_features(video_path, start, end,fps):
    img_features = {}
    clip = mp.VideoFileClip(video_path)
    video = clip.subclip(start, end)
    video = video.set_fps(fps)
    total_duration = video.duration
    fps = video.fps
    frame_count = total_duration * fps
    subclip = video.iter_frames()
    count = 0
    with tqdm(total=frame_count, desc="Extracting Features") as pbar:
        while True:
            try:
                frame = next(subclip)
                count += 1
                img = frame
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (224,224))
                img = torch.Tensor(img)
                img = img.permute(2, 0, 1)
                img = img.unsqueeze(0).to(device)
                with torch.no_grad():
                    features = model(img).view(512)
                if torch.cuda.is_available():
                    features = features.cpu().detach().numpy()
                else:
                    features = features.detach().numpy()
                img_features[count] = features
                pbar.update(1)
                if count % 500 == 0:
                    pbar.set_postfix({"Processed frames": count})
            except StopIteration:
                break
        return [img_features, video, total_duration]

def eratosthenis(img_features, n):
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
    
def optimal_k(feature_vectors):
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

def cluster(data):
    key_frames = []
    k = optimal_k(data)
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

def generate_keyframes(img_features):
    sets = eratosthenis(img_features, 5)
    key_frames = []
    final_kf = []
    for idx, set in enumerate(sets):
        var = cluster(set)
        key_frames.extend(var)
    for i in key_frames:
        final_kf.append(frame_mapper(img_features, i))
    final_kf.sort()
    return final_kf


def frame_mapper(dictionary, value):
    for key, val in dictionary.items():
        if np.array_equal(val, value):
            return key

def boundary_determination(feature_vectors_dict, keyframes, initial_event_boundary_threshold, percentage):
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


def output_summary(video_clip, frame_ranges, output_path,fps):
    video_clip = video_clip.set_fps(fps)
    fps = video_clip.fps
    clips = []
    for start_frame, end_frame in frame_ranges:
        clip = video_clip.subclip(
            start_frame / fps, end_frame / fps)
        clips.append(clip)
    final_clip = mp.concatenate_videoclips(clips)
    final_clip.write_videofile(
        output_path, codec="libx264", audio_codec="aac")

def save_details(file_path, video_details):
    with open(file_path, mode="w+", newline="") as csvfile:
        writer = csv.writer(csvfile)
        header = [
            "Video Name",
            "Total Frames",
            "Keyframes",
        ]
        writer.writerow(header)
        writer.writerow(video_details)

def time_duration(video):
    clip = mp.VideoFileClip(video)
    return int(clip.duration)


def time_boundary_determination(feature_vectors_dict, keyframes, seconds, fps):
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

# video_path = "Short-1min.mp4"
# videoName = "45mins"

# img_features,video = extract_features(video_path,0,60)
# keyframes = generate_keyframes(img_features)

# event_boundary_threshold = 0.99
# percentage = 100

# event_boundaries,ebt = boundary_determination(img_features, keyframes, event_boundary_threshold, percentage)

# output_path=videoName+"_summary.mp4"
# output_summary(video_path, event_boundaries,output_path, videoName)