import torch
import matplotlib.pyplot as plt
import numpy as np
import skimage.io
from PIL import Image
left = Image.open('/mnt/nas/puyuechuan/Scene_Flow_Datasets/frames_finalpass/eating_naked_camera2_x2/left/0077.png').convert('RGB')
left = np.array(left)
w, h, c = left.shape
print(left,left.shape)
crop_w, crop_h = 960, 512
left_img = left.crop((w - crop_w, h - crop_h, w, h))
print(left_img.size)
fn = "/home/xgw/gwc_attention/checkpoints/kitti_test/attention_com_sceneflow_left/"
skimage.io.imsave(fn, left_img)