# from __future__ import print_function, division
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
from models import __models__, model_loss_train, model_loss_test
from utils import *
from torch.utils.data import DataLoader
import gc
# from apex import amp
import cv2

cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(description='Guard-Net')
parser.add_argument('--model', default='CoEx', help='select a model structure', choices=__models__.keys())
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')
parser.add_argument('--dataset', default='sceneflow', help='dataset name', choices=__datasets__.keys())
parser.add_argument('--datapath', default="", help='data path')
parser.add_argument('--testlist',default='./filenames/sceneflow_test.txt', help='testing list')
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--loadckpt', default='',help='load the weights from a specific checkpoint')
# parse arguments, set seeds
args = parser.parse_args()


# dataset, dataloader
StereoDataset = __datasets__[args.dataset]
test_dataset = StereoDataset(args.datapath, args.testlist, False)
TestImgLoader = DataLoader(test_dataset, args.test_batch_size, shuffle=False, num_workers=24, drop_last=False)

# model, optimizer
model = __models__[args.model](args.maxdisp)
model = nn.DataParallel(model)
model.cuda()
total = sum([param.nelement() for param in model.parameters()])
print("Number of parameter: %.2f Millions" % (total/1e6))
# print_param(model)
# load parameters
print("loading model {}".format(args.loadckpt))
state_dict = torch.load(args.loadckpt)
model.load_state_dict(state_dict['model'])

save_dir = './disparity_results/'


def test():
    os.makedirs(save_dir, exist_ok=True)
    avg_test_scalars = AverageMeterDict()
    for batch_idx, sample in enumerate(TestImgLoader):
        torch.cuda.synchronize()
        start_time = time.time()
        disp_est_np = tensor2numpy(test_sample(sample))
        torch.cuda.synchronize()
        print('Iter {}/{}, time = {:3f}'.format(batch_idx, len(TestImgLoader),
                                                time.time() - start_time))
        left_filenames = sample["left_filename"]

        for disp_est, fn in zip(disp_est_np, left_filenames):
            assert len(disp_est.shape) == 2
            disp_est = np.array(disp_est, dtype=np.float32)
            fn = os.path.join(save_dir, fn.split('/')[-4]+fn.split('/')[-3]+fn.split('/')[-1])
            print("saving to", fn, disp_est.shape)
            disp_est_uint = np.round(disp_est * 256).astype(np.uint16)
            # skimage.io.imsave(fn, disp_est_uint)
            cv2.imwrite(fn, cv2.applyColorMap(cv2.convertScaleAbs(disp_est_uint, alpha=0.01),cv2.COLORMAP_JET))



# test one sample
@make_nograd_func
def test_sample(sample, compute_metrics=True):
    model.eval()
    disp_ests, _ = model(sample['left'].cuda(), sample['right'].cuda())
    return disp_ests[-1]

if __name__ == '__main__':
    test()