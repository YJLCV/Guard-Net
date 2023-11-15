import os
import random
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import cv2
from datasets.data_io import get_transform, read_all_lines, pfm_imread
import torchvision.transforms as transforms
import torch
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import matplotlib.pyplot as plt




def load_image(filename):
    return Image.open(filename).convert('RGB')



def read_HR():
    left_img = load_image('/home/xgw/ACV_AP/HR_VS_imgs/left.png')
    right_img = load_image('/home/xgw/ACV_AP/HR_VS_imgs/right.png')
    w, h = left_img.size
    h_pad = (32-h%32)%32
    w_pad = (32-w%32)%32
    processed = get_transform()
    left_img = processed(left_img)
    right_img = processed(right_img)
    left_img = np.lib.pad(left_img, ((0, 0), (0, h_pad), (0, w_pad)), mode='constant', constant_values=0)
    right_img = np.lib.pad(right_img, ((0, 0), (0, h_pad), (0, w_pad)), mode='constant', constant_values=0)

    return {"left": left_img,
            "right": right_img,
            "h": h,
            "w": w}