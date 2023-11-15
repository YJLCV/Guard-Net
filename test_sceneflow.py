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
parser.add_argument('--datapath', default="/home/yjl/dataset/SceneFlow/", help='data path')
parser.add_argument('--testlist',default='./filenames/sceneflow_test.txt', help='testing list')
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--loadckpt', default='./yjl_models/sceneflow_0.47.ckpt',help='load the weights from a specific checkpoint')
# parse arguments, set seeds
args = parser.parse_args()

def print_param(model):
    """
    print number of parameters in the model
    """
    feature_extraction = sum(p.numel() for n, p in model.named_parameters() if 'feature_extraction' in n and p.requires_grad)
    concatconv = sum(p.numel() for n, p in model.named_parameters() if 'concatconv' in n and p.requires_grad)
    n_parameters = feature_extraction+concatconv
    print("Number of parameter in feature_extraction: %.2f Millions" % (n_parameters/1e6))

    tlmodule = sum(p.numel() for n, p in model.named_parameters() if 'tlmodule' in n and p.requires_grad)
    br_module = sum(p.numel() for n, p in model.named_parameters() if 'br_module' in n and p.requires_grad)
    n_parameters = tlmodule+br_module
    print("Number of parameter in corner_pooling: %.2f Millions" % (n_parameters/1e6))

    dres = sum(p.numel() for n, p in model.named_parameters() if 'dres' in n and p.requires_grad)
    dres1 = sum(p.numel() for n, p in model.named_parameters() if 'dres1' in n and p.requires_grad)
    dres2 = sum(p.numel() for n, p in model.named_parameters() if 'dres2' in n and p.requires_grad)
    dres3 = sum(p.numel() for n, p in model.named_parameters() if 'dres3' in n and p.requires_grad)
    classif0 = sum(p.numel() for n, p in model.named_parameters() if 'classif0' in n and p.requires_grad)
    classif1 = sum(p.numel() for n, p in model.named_parameters() if 'classif1' in n and p.requires_grad)
    classif2 = sum(p.numel() for n, p in model.named_parameters() if 'classif2' in n and p.requires_grad)
    n_parameters = dres2+dres3
    # n_parameters = dres+dres1+dres2+dres3+classif0+classif1+classif2
    print("Number of parameter in cost_aggregation: %.2f Millions" % (n_parameters/1e6))



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

save_dir = './error_unmap_results/'


def test():
    avg_test_scalars = AverageMeterDict()
    # os.makedirs(save_dir, exist_ok=True)
    for batch_idx, sample in enumerate(TestImgLoader):   
        left_filenames = sample["left_filename"][0]
        start_time = time.time()
        loss, scalar_outputs, image_outputs, unmap = test_sample(sample)
        avg_test_scalars.update(scalar_outputs)
        
        del scalar_outputs
        error_map = image_outputs["errormap"][0].squeeze().data.cpu().numpy().transpose(1, 2, 0)
        error_map = error_map[:, :, [2, 1, 0]] * 255.0
        fn = os.path.join(save_dir, left_filenames.split('/')[-4]+left_filenames.split('/')[-3]+left_filenames.split('/')[-1])
        fn_numap = os.path.join(save_dir, left_filenames.split('/')[-4]+left_filenames.split('/')[-3]+left_filenames.split('/')[-1].split(".")[0]+"_nu_map.png")
        cv2.imwrite(fn, error_map)


        un_map = unmap[0].squeeze(1).data.cpu().numpy().transpose(1, 2, 0)
        un_map = cv2.cvtColor((un_map*255.0), cv2.COLOR_GRAY2BGR)
        # un_map = un_map[:, :, [2, 1, 0]] * 255.0
        # import pdb
        # pdb.set_trace()
        cv2.imwrite(fn_numap, un_map)

        # print(fn.split('/')[-1] + ":", avg_test_scalars.mean())

        
    avg_test_scalars = avg_test_scalars.mean()
    print("All Dataset: ", avg_test_scalars)
    print('> EPE: ', avg_test_scalars['EPE'][0])
    print('> D1: ', avg_test_scalars['D1'][0])
    print('> 1px: ', avg_test_scalars['Thres1'][0])
    print('> 2px: ', avg_test_scalars['Thres2'][0])
    print('> 3px: ', avg_test_scalars['Thres3'][0])



# test one sample
@make_nograd_func
def test_sample(sample, compute_metrics=True):
    model.eval()
    #model.train()

    # imgL, imgR, disp_gt, gradient_map, disp_gt_low = sample['left'], sample['right'], sample['disparity'], sample['gradient_map'], sample['disparity_low']
    imgL, imgR, disp_gt, disp_gt_low = sample['left'], sample['right'], sample['disparity'], sample['disparity_low']
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    disp_gt = disp_gt.cuda()
    disp_gt_low = disp_gt_low.cuda()
    mask = (disp_gt < args.maxdisp) & (disp_gt > 0)
    mask_low = (disp_gt_low < args.maxdisp) & (disp_gt_low > 0)
    import time
    start_time = time.time()
    disp_ests, un_maps = model(imgL, imgR)
    end_time = time.time()

    print("time: ", end_time-start_time)
    disp_gts = [disp_gt]
    masks = [mask]
    loss = model_loss_test(disp_ests, disp_gts, masks)
    
    image_outputs = {"disp_est": disp_ests, "disp_gt": disp_gt, "imgL": imgL, "imgR": imgR}
    image_outputs["errormap"] = [disp_error_image_func.apply(disp_est, disp_gt) for disp_est in disp_ests]
    scalar_outputs = {"loss": loss}
    scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests]
    scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests]
    scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests]

    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs, un_maps

if __name__ == '__main__':
    test()
# run
# python test_sceneflow.py >> cornerv2.log 