import cv2
from matplotlib import pyplot as plt
from typing import List, Dict
import numpy as np
import json

def generate_image(arr : List[bool]):
    line = np.array([255 if x else 0 for x in arr],dtype=np.uint8)
    image = np.vstack([line for i in range(60)])
    print(image.shape)
    return image

def load_values() -> Dict[str,List[bool]]:
    with open("debug.json") as f:
        data = json.load(f)
    for k in data.keys():
        print(f"Avg of {k}: {np.average(data[k])}")
        data[k] = list([True if x == 1 else False for x in data[k]])
    return data

if __name__ == '__main__':
    data = load_values()
    plt.gray()
    for i,k in enumerate(data.keys()):
        plt.subplot(len(data),1,i+1)
        plt.imshow(cv2.resize(generate_image(data[k]),(300,60)))
        plt.title(k)
    plt.show()
    
    