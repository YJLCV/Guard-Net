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

#                    _ooOoo_
#                   o8888888o
#                   88" . "88
#                   (| -_- |)
#                   O\  =  /O
#                ____/`---'\____
#              .'  \\|     |//  `.
#             /  \\|||  :  |||//  \
#            /  _||||| -:- |||||-  \
#            |   | \\\  -  /// |   |
#            | \_|  ''\---/''  |   |
#            \  .-\__  `-`  ___/-. /
#          ___`. .'  /--.--\  `. . __
#       ."" '<  `.___\_<|>_/___.'  >'"".
#      | | :  `- \`.;`\ _ /`;.`/ - ` : | |
#      \  \ `-.   \_ __\ /__ _/   .-` /  /
# ======`-.____`-.___\_____/___.-`____.-'======
#                    \=---=/
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#           佛祖保佑   state of the art

cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(description='Guard-Net')
parser.add_argument('--model', default='CoEx', help='select a model structure', choices=__models__.keys())
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')

# parser.add_argument('--dataset', default='sceneflow', help='dataset name', choices=__datasets__.keys())
# parser.add_argument('--datapath', default="/data/sceneflow/", help='data path')
# parser.add_argument('--trainlist', default='./filenames/train_scene_flow.txt', help='training list')
# parser.add_argument('--testlist',default='./filenames/sceneflow_test.txt', help='testing list')

parser.add_argument('--dataset', default='kitti', help='dataset name', choices=__datasets__.keys())
parser.add_argument('--kitti15_datapath', default='', help='data path')
parser.add_argument('--kitti12_datapath', default='', help='data path')
parser.add_argument('--trainlist', default='./filenames/kitti12_all.txt', help='training list')
parser.add_argument('--testlist',default='./filenames/kitti12_val.txt', help='testing list')

parser.add_argument('--lr', type=float, default=0.001, help='base learning rate')
#parser.add_argument('--lr', type=float, default=0.000015625, help='base learning rate')
parser.add_argument('--batch_size', type=int, default=10, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=16, help='testing batch size')
parser.add_argument('--epochs', type=int, default=500, help='number of epochs to train')
# parser.add_argument('--lrepochs',default="15,20,24,27,29,31:2", type=str,  help='the epochs to decay lr: the downscale rate')
parser.add_argument('--lrepochs',default="300:10", type=str,  help='the epochs to decay lr: the downscale rate')
# parser.add_argument('--attention_weights_only', default=False, type=str,  help='only train attention weights')

# parser.add_argument('--logdir',default='/data/xgw/ACV_AP/acv_vap_semantic_topk/', help='the directory to save logs and checkpoints')
parser.add_argument('--logdir',default='./yjl_0.47_kitti12/', help='the directory to save logs and checkpoints')
#parser.add_argument('--logdir',default='/home/xgw/gwc_attention/checkpoints/attention_kitti/test/', help='the directory to save logs and checkpoints')
parser.add_argument('--loadckpt', default='./yjl_0.47_kitti1215/coex_kitti12+15_499_epoch.ckpt',help='load the weights from a specific checkpoint')
# parser.add_argument('--loadckpt', default='/data/xgw/ACV_AP/acv_vap_new_variance/acv_vap_variance_65.ckpt',help='load the weights from a specific checkpoint')
# parser.add_argument('--loadckpt', default='./pretrained_model_sceneflow_48.ckpt',help='load the weights from a specific checkpoint')
parser.add_argument('--resume', action='store_true', help='continue training the model')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

parser.add_argument('--summary_freq', type=int, default=20, help='the frequency of saving summary')
parser.add_argument('--save_freq', type=int, default=50, help='the frequency of saving checkpoint')

# parse arguments, set seeds
args = parser.parse_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
os.makedirs(args.logdir, exist_ok=True)

# create summary logger
print("creating new summary file")
logger = SummaryWriter(args.logdir)

# dataset, dataloader
StereoDataset = __datasets__[args.dataset]
train_dataset = StereoDataset(args.kitti15_datapath, args.kitti12_datapath, args.trainlist, True)
test_dataset = StereoDataset(args.kitti15_datapath, args.kitti12_datapath, args.testlist, False)
TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=24, drop_last=True)
TestImgLoader = DataLoader(test_dataset, args.test_batch_size, shuffle=False, num_workers=24, drop_last=False)

# model, optimizer
model = __models__[args.model](args.maxdisp)
# model = nn.DataParallel(model)
# model.cuda()
# optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
# #optimizer = optim.SGD(model.parameters(), lr = args.lr, momentum=0.9)
# model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
model = nn.DataParallel(model)
model.cuda()
total = sum([param.nelement() for param in model.parameters()])
print("Number of parameter: %.2f Millions" % (total/1e6))
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
# optimizer = optim.SGD(model.parameters(), lr = args.lr, momentum=0.9)
# load parameters
start_epoch = 0
if args.resume:
    # find all checkpoints file and sort according to epoch id
    all_saved_ckpts = [fn for fn in os.listdir(args.logdir) if fn.endswith(".ckpt")]
    all_saved_ckpts = sorted(all_saved_ckpts, key=lambda x: int(x.split('_')[2]))
    # use the latest checkpoint file
    loadckpt = os.path.join(args.logdir, all_saved_ckpts[-1])
    print("loading the lastest model in logdir: {}".format(loadckpt))
    state_dict = torch.load(loadckpt)
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    start_epoch = state_dict['epoch'] + 1
elif args.loadckpt:
    # load the checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)

    #print('####model',state_dict['model'].keys())

    model_dict = model.state_dict()
    # print(model_dict)
    pre_dict = {k: v for k, v in state_dict['model'].items() if k in model_dict}
    #print('#####model',pre_dict.keys())
    model_dict.update(pre_dict) 
    # model.load_state_dict(state_dict['model'])
    model.load_state_dict(model_dict)
print("start at epoch {}".format(start_epoch))


def train():
    #for epoch_idx in range(start_epoch, args.epochs):
    for epoch_idx in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch_idx, args.lr, args.lrepochs)

        # training
        #aaa = 1
        for batch_idx, sample in enumerate(TrainImgLoader):

            # if batch_idx == 0:
            #     break
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            start_time = time.time()
            do_summary = global_step % args.summary_freq == 0
            #loss, scalar_outputs, image_outputs = train_sample(sample, compute_metrics=do_summary)
            loss,scalar_outputs= train_sample(sample, compute_metrics=do_summary)
            #print('bn weight')
            #print(model.state_dict().keys())
            #print(model.state_dict()['module.feature_extraction.firstconv.4.1.running_var'])
            #print(model.state_dict()['module.feature_extraction.aspp_layer.bn_conv_1x1_3.running_mean'])
            #print(model.state_dict()['module.feature_extraction.aspp_layer.bn_conv_1x1_3.running_var'])
            if do_summary:
                save_scalars(logger, 'train', scalar_outputs, global_step)
               # save_images(logger, 'train', image_outputs, global_step)
            #del scalar_outputs, image_outputs
            del scalar_outputs
            print('Epoch {}/{}, Iter {}/{}, train loss = {:.3f}, time = {:.3f}'.format(epoch_idx, args.epochs,
                                                                                       batch_idx,
                                                                                       len(TrainImgLoader), loss,
                                                                                       time.time() - start_time))

            

            
        # saving checkpoints

        if (epoch_idx + 1) % args.save_freq == 0:
            checkpoint_data = {'epoch': epoch_idx, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
            #id_epoch = (epoch_idx + 1) % 100
            # torch.save(checkpoint_data, "{}/checkpoint_{:0>6}.ckpt".format(args.logdir, epoch_idx))
            torch.save(checkpoint_data, "{}/coex_kitti12+15_{}_epoch.ckpt".format(args.logdir, epoch_idx))
        gc.collect()

        # # testing
        # avg_test_scalars = AverageMeterDict()
        # for batch_idx, sample in enumerate(TestImgLoader):

        #     global_step = len(TestImgLoader) * epoch_idx + batch_idx
        #     start_time = time.time()
        #     # do_summary = global_step % args.summary_freq == 0
        #     do_summary = global_step % 1 == 0
        #     # loss, scalar_outputs, image_outputs = test_sample(sample, compute_metrics=do_summary)
        #     loss, scalar_outputs = test_sample(sample, compute_metrics=do_summary)
        #     #print('bn weight')
        #     #print(model.state_dict()['module.feature_extraction.aspp_layer.bn_conv_1x1_3.running_mean'])
        #     #print(model.state_dict()['module.feature_extraction.aspp_layer.bn_conv_1x1_3.running_var'])
        #     if do_summary:
        #        save_scalars(logger, 'test', scalar_outputs, global_step)
        #        # save_images(logger, 'test', image_outputs, global_step)
        #     avg_test_scalars.update(scalar_outputs)
        #     #del scalar_outputs, image_outputs
        #     print('Epoch {}/{}, Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(epoch_idx, args.epochs,
        #                                                                              batch_idx,
        #                                                                              len(TestImgLoader), loss,
        #                                                                              time.time() - start_time))
        # avg_test_scalars = avg_test_scalars.mean()
        # save_scalars(logger, 'fulltest', avg_test_scalars, len(TrainImgLoader) * (epoch_idx + 1))
        # print("avg_test_scalars", avg_test_scalars)
        # gc.collect()


# train one sample
def train_sample(sample, compute_metrics=False):
    model.train()
    #print('sample',sample.keys())

    # imgL, imgR, disp_gt, gradient_map, disp_gt_low = sample['left'], sample['right'], sample['disparity'], sample['gradient_map'], sample['disparity_low']
    imgL, imgR, disp_gt, disp_gt_low = sample['left'], sample['right'], sample['disparity'], sample['disparity_low']
    # imgL_np = tensor2numpy(imgL).transpose(0,2,3,1)
    # print(imgL_np[:,].shape)
    # dx_imgL[:,] = cv2.Sobel(imgL_np[:,],cv2.CV_32F,1,0,ksize=3)
    # dy_imgL[:,] = cv2.Sobel(imgL_np[:,],cv2.CV_32F,0,1,ksize=3)
    # dxy_imgL=np.sqrt(np.sum(np.square(dx_imgL),axis=-1)+np.sum(np.square(dy_imgL),axis=-1))
    # mask_img = (dxy_imgL > np.percentile(dxy_imgL,80)) * 1.0
    #imgL, imgR, disp_gt, mask_edge0, mask_edge1, mask_edge2, mask_smooth = sample['left'], sample['right'], sample['disparity'], \
    #                                                                          sample['mask_edge0'], sample['mask_edge1'], sample['mask_edge2'], sample['mask_smooth']

    #print('1111',imgL.size())
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    disp_gt = disp_gt.cuda()
    disp_gt_low = disp_gt_low.cuda()
    # gradient_map = gradient_map.cuda()
    #mask_edge = mask_edge.cuda()
    #mask_edge0 = mask_edge0.cuda()
    #mask_edge1 = mask_edge1.cuda()
    #mask_edge2 = mask_edge2.cuda()
    #mask_smooth = mask_smooth.cuda()

    optimizer.zero_grad()

    #disp_ests,masks_uncert = model(imgL, imgR)
    disp_ests = model(imgL, imgR)
    mask = (disp_gt < args.maxdisp) & (disp_gt > 0)
    mask_low = (disp_gt_low < args.maxdisp) & (disp_gt_low > 0)
    masks = [mask, mask_low, mask, mask_low]
    #print('####2111',mask_low.shape)
    #mask_front = (disp_gt < args.maxdisp) & (disp_gt > 20)
    
    # masks = [(mask_smooth&mask).type(torch.bool), (mask_edge&mask).type(torch.bool), \
    #             mask.type(torch.bool), mask.type(torch.bool), mask.type(torch.bool)]
    # masks = [mask.type(torch.bool), mask.type(torch.bool), \
    #             mask.type(torch.bool), mask_low.type(torch.bool), mask_low.type(torch.bool), mask_low.type(torch.bool)]
    disp_gts = [disp_gt, disp_gt_low, disp_gt, disp_gt_low] 
    # masks = [mask.type(torch.bool),  \
    #             mask.type(torch.bool), mask.type(torch.bool), mask.type(torch.bool)]
    # masks = [(mask_smooth&mask).type(torch.bool), (mask_edge&mask).type(torch.bool), \
    #         (mask_smooth&mask).type(torch.bool), (mask_edge&mask).type(torch.bool), mask.type(torch.bool), mask.type(torch.bool)]
    # print((mask_smooth&mask).type(torch.bool).dtype)
    # masks = [mask_smooth&mask, mask_edge0&mask, mask_edge1&mask, mask_edge2&mask, mask, mask, mask]
    # masks = [mask_smooth&mask, mask_edge&mask]
    # masks = [mask_smooth&mask, mask_edge0&mask, mask_edge1&mask, mask_edge2&mask, mask_smooth&mask, mask_edge0&mask, mask_edge1&mask, mask_edge2&mask, mask, mask]
    #masks = [mask_smooth&mask, mask_edge0&mask, mask_edge1&mask, mask_edge2&mask, \
    #        mask_smooth&mask, mask_edge0&mask, mask_edge1&mask, mask_edge2&mask, mask, mask]
    # masks = [mask_smooth&mask, mask_edge0&mask, mask_edge1&mask, mask_edge2&mask, \
    #         mask_smooth&mask, mask_edge0&mask, mask_edge1&mask, mask_edge2&mask,\
    #         mask_smooth&mask, mask_edge0&mask, mask_edge1&mask, mask_edge2&mask, mask]
    #loss = model_loss_uncertainty_masks(disp_ests, disp_gt, mask, masks,masks_uncert)
    loss = model_loss_train(disp_ests, disp_gts, masks)
    #print('loss',loss)

    scalar_outputs = {"loss": loss}
    # disp_ests.append(disp_ests[0]*mask_smooth.type(torch.cuda.FloatTensor) + disp_ests[1]*mask_edge.type(torch.cuda.FloatTensor))
    # image_outputs = {"disp_est": disp_ests, "disp_gt": disp_gt, "imgL": imgL, "imgR": imgR}
    if compute_metrics:
        with torch.no_grad():
            # masks = [mask_smooth, mask_edge, mask*1.0]
            # image_outputs["errormap"] = [disp_error_image_func()(disp_est, disp_gt) for disp_est in disp_ests]
            # scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
            # scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
            # scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests]
            # scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests]
            # scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests]
            scalar_outputs["EPE"] = [EPE_metric_mask(disp_est, disp_gt, mask, mask_) for disp_est,disp_gt, mask, mask_ in zip(disp_ests,disp_gts, masks, masks)]
            scalar_outputs["D1"] = [D1_metric_mask(disp_est, disp_gt, mask, mask_) for disp_est, disp_gt, mask, mask_ in zip(disp_ests, disp_gts, masks, masks)]
            # scalar_outputs["Thres1"] = [Thres_metric_mask(disp_est, disp_gt, mask, 1.0, mask_) for disp_est, mask_ in zip(disp_ests, masks)]
            # scalar_outputs["Thres2"] = [Thres_metric_mask(disp_est, disp_gt, mask, 2.0, mask_) for disp_est, mask_ in zip(disp_ests, masks)]
            # scalar_outputs["Thres3"] = [Thres_metric_mask(disp_est, disp_gt, mask, 3.0, mask_) for disp_est, mask_ in zip(disp_ests, masks)]
    loss.backward()
    # with amp.scale_loss(loss, optimizer) as scaled_loss:
    #     scaled_loss.backward()
    optimizer.step()

    #return tensor2float(loss), tensor2float(scalar_outputs), image_outputs
    return tensor2float(loss), tensor2float(scalar_outputs)


# test one sample
@make_nograd_func
def test_sample(sample, compute_metrics=True):
    model.eval()
    #model.train()

    # imgL, imgR, disp_gt, gradient_map, disp_gt_low = sample['left'], sample['right'], sample['disparity'], sample['gradient_map'], sample['disparity_low']
    imgL, imgR, disp_gt = sample['left'], sample['right'], sample['disparity']
    # imgL_np = tensor2numpy(imgL).squeeze().transpose(1,2,0)
    # dx_imgL = cv2.Sobel(imgL_np,cv2.CV_32F,1,0,ksize=3)
    # dy_imgL = cv2.Sobel(imgL_np,cv2.CV_32F,0,1,ksize=3)
    # dxy_imgL=np.sqrt(np.sum(np.square(dx_imgL),axis=-1)+np.sum(np.square(dy_imgL),axis=-1))
    # mask_img = (dxy_imgL > np.percentile(dxy_imgL,80)) * 1.0

    #imgL, imgR, disp_gt, mask_edge0, mask_edge1, mask_edge2, mask_smooth = sample['left'], sample['right'], sample['disparity'], \
    #                                                                           sample['mask_edge0'], sample['mask_edge1'], sample['mask_edge2'], sample['mask_smooth']

    imgL = imgL.cuda()
    imgR = imgR.cuda()
    disp_gt = disp_gt.cuda()
    # disp_gt_low = disp_gt_low.cuda()
    # mask_low = (disp_gt_low < args.maxdisp) & (disp_gt_low > 0)
    #mask_edge = mask_edge.cuda()
    #mask_edge0 = mask_edge0.cuda()
    #mask_edge1 = mask_edge1.cuda()
    #mask_edge2 = mask_edge2.cuda()
    #mask_smooth = mask_smooth.cuda()
    

    #disp_ests = model(imgL, imgR)
    mask = (disp_gt < args.maxdisp) & (disp_gt > 0)
    #print(mask.type(),mask_edge.type())
    disp_ests = model(imgL, imgR)
    disp_gts = [disp_gt]
    masks = [mask]
    
    # masks = [mask_smooth&mask, mask_edge&mask]
    loss = model_loss_test(disp_ests, disp_gts, masks)
    #print('####11111',loss)

    scalar_outputs = {"loss": loss}
    # disp_ests.append(disp_ests[0]*mask_smooth.type(torch.cuda.FloatTensor) + disp_ests[1]*mask_edge.type(torch.cuda.FloatTensor))

    # image_outputs = {"disp_est": disp_ests, "disp_gt": disp_gts, "imgL": imgL, "imgR": imgR}

    # masks = [mask_smooth, mask_edge, (mask*1.0).type(torch.cuda.FloatTensor)]

    scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    # scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests]
    # scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests]
    # scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests]

    # scalar_outputs["EPE"] = [EPE_metric_mask(disp_est, disp_gt, mask, mask_) for disp_est, disp_gt, mask, mask_ in zip(disp_ests, disp_gts, masks, masks)]
    # scalar_outputs["D1"] = [D1_metric_mask(disp_est, disp_gt, mask, mask_) for disp_est, disp_gt, mask, mask_ in zip(disp_ests, disp_gts, masks, masks)]
    # scalar_outputs["Thres1"] = [Thres_metric_mask(disp_est, disp_gt, mask, 1.0, mask_) for disp_est, disp_gt, mask, mask_ in zip(disp_ests, disp_gts, masks, masks)]
    # scalar_outputs["Thres2"] = [Thres_metric_mask(disp_est, disp_gt, mask, 2.0, mask_) for disp_est, disp_gt, mask, mask_ in zip(disp_ests, disp_gts, masks, masks)]
    # scalar_outputs["Thres3"] = [Thres_metric_mask(disp_est, disp_gt, mask, 3.0, mask_) for disp_est, disp_gt, mask, mask_ in zip(disp_ests, disp_gts, masks, masks)]

    # if compute_metrics:
    #     image_outputs["errormap"] = [disp_error_image_func()(disp_est, disp_gt) for disp_est in disp_ests]

    # return tensor2float(loss), tensor2float(scalar_outputs), image_outputs
    return tensor2float(loss), tensor2float(scalar_outputs)


if __name__ == '__main__':
    train()
