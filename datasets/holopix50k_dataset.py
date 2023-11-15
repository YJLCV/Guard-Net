import os
import random
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2
from datasets.data_io import get_transform, read_all_lines, pfm_imread
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt



def listfiles(filepath):

    #filepath = './data_demo/KITTI_raw/2011_09_26/2011_09_26_drive_0005_sync/'
    left_fold = 'left'
    right_fold = 'right'
    datadir_left = '{}/{}'.format(filepath, left_fold)
    datadir_right = '{}/{}'.format(filepath, right_fold)
    img_names = os.listdir(datadir_left)
    # img_names.sort()
    left_imgs = []
    right_imgs = []
    for i in range(len(img_names)):
        left_imgs.append(os.path.join(datadir_left, img_names[i]))
        right_img_names = img_names[i][:-8] + 'right.jpg'
        right_imgs.append(os.path.join(datadir_right, right_img_names))

    return left_imgs, right_imgs


class ImageLoader(Dataset):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        left_images = [x[0] for x in splits]
        right_images = [x[1] for x in splits]
        if len(splits[0]) == 2:  # ground truth not available
            return left_images, right_images, None
        else:
            disp_images = [x[2] for x in splits]
            return left_images, right_images, disp_images

    def load_image(self, filename):
        return Image.open(filename).convert('RGB')

    def load_disp(self, filename):
        data = Image.open(filename)
        data = np.array(data, dtype=np.float32) / 256.
        return data

    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, index):
        batch = dict()
        left = self.left[index]
        right = self.right[index]
        left_img = self.load_image(left)
        right_img = self.load_image(right)
        w, h = left_img.size
        h_pad = (32-h%32)%32
        w_pad = (32-w%32)%32
        # h, w, _ = left_img.shape
        processed = get_transform()

        left_img_p = processed(left_img).numpy()
        right_img_p = processed(right_img).numpy()
        left_img_p = np.lib.pad(left_img_p, ((0, 0), (0, h_pad), (0, w_pad)), mode='constant', constant_values=0)
        right_img_p = np.lib.pad(right_img_p, ((0, 0), (0, h_pad), (0, w_pad)), mode='constant', constant_values=0)

        left_img_ = np.transpose(left_img, (2, 0, 1)).astype(np.float32)
        right_img_ = np.transpose(right_img, (2, 0, 1)).astype(np.float32)
        batch['imgL'], batch['imgR'] = left_img_p, right_img_p
        batch['imgLRaw'], batch['imgRRaw'] = left_img_, right_img_
        batch['h'], batch['w'] = h, w
        batch['left_filename'] = self.left[index]

        return batch

    def __len__(self):
        return len(self.left)
