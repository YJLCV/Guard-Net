from __future__ import print_function, division
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torchvision.utils as vutils
import torch.nn.functional as F
import numpy as np
import time
from tensorboardX import SummaryWriter
from datasets import __datasets__
from models import __models__
from utils import *
from torch.utils.data import DataLoader
import gc
import skimage
import skimage.io
import cv2
# cudnn.benchmark = True

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(description='Attention Concatenation Volume for Accurate and Efficient Stereo Matching (ACVNet)')
parser.add_argument('--model', default='CoEx', help='select a model structure', choices=__models__.keys())
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')
parser.add_argument('--dataset', default='kitti', help='dataset name', choices=__datasets__.keys())
parser.add_argument('--kitti15_datapath', default='/home/yjl/dataset/KITTI/2015/', help='data path')
parser.add_argument('--kitti12_datapath', default='/home/yjl/dataset/KITTI/2012/', help='data path')
#############  测试时候需要修改得地方  #############
parser.add_argument('--testlist', default='./filenames/kitti12_test.txt', help='testing list')
parser.add_argument('--loadckpt', default='./yjl_models/kitti12.ckpt',help='load the weights from a specific checkpoint')
#############  测试时候需要修改得地方  #############
parser.add_argument('--attention_weights_only', default=False, type=str,  help='only train attention weights')
# parse arguments
args = parser.parse_args()

# dataset, dataloader
StereoDataset = __datasets__[args.dataset]
test_dataset = StereoDataset(args.kitti15_datapath, args.kitti12_datapath, args.testlist, False)
TestImgLoader = DataLoader(test_dataset, 1, shuffle=False, num_workers=4, drop_last=False)

# model, optimizer
model = __models__[args.model](args.maxdisp)
model = nn.DataParallel(model)
model.cuda()

#load parameters
print("loading model {}".format(args.loadckpt))
state_dict = torch.load(args.loadckpt)
model.load_state_dict(state_dict['model'])


save_dir = './output/15/disp_0/'


def test():
    os.makedirs(save_dir, exist_ok=True)
    for batch_idx, sample in enumerate(TestImgLoader):
        # torch.cuda.synchronize()
        start_time = time.time()
        disp_est_np = tensor2numpy(test_sample(sample))
        # torch.cuda.synchronize()
        print('Iter {}/{}, time = {:3f}'.format(batch_idx, len(TestImgLoader),
                                                time.time() - start_time))
        top_pad_np = tensor2numpy(sample["top_pad"])
        right_pad_np = tensor2numpy(sample["right_pad"])
        left_filenames = sample["left_filename"]

        for disp_est, top_pad, right_pad, fn in zip(disp_est_np, top_pad_np, right_pad_np, left_filenames):
            assert len(disp_est.shape) == 2
            disp_est = np.array(disp_est[top_pad:, :-right_pad], dtype=np.float32)
            fn = os.path.join(save_dir, fn.split('/')[-1])
            print("saving to", fn, disp_est.shape)
            disp_est_uint = np.round(disp_est * 256).astype(np.uint16)
            skimage.io.imsave(fn, disp_est_uint)
            # cv2.imwrite(fn, cv2.applyColorMap(cv2.convertScaleAbs(disp_est_uint, alpha=0.01),cv2.COLORMAP_JET))


# test one sample
@make_nograd_func
def test_sample(sample):
    
    start_time = time.time()
    model.eval()
    disp_ests, _ = model(sample['left'].cuda(), sample['right'].cuda())
    end_time = time.time()
    print("time: ", end_time-start_time)
    return disp_ests[-1]


if __name__ == '__main__':
    test()
