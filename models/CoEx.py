from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from models.submodule import *
import math
import gc
import time
import timm
from models.axial_transformer import BlockAxial, my_Block_2
from models.aspp import *
from models.uncertainty_adjustment import *



class SubModule(nn.Module):
    def __init__(self):
        super(SubModule, self).__init__()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Feature(SubModule):
    def __init__(self):
        super(Feature, self).__init__()
        pretrained =  True
        # online 
        model = timm.create_model('mobilenetv2_100', pretrained=pretrained, features_only=True)
        
        # offline
        # model = timm.create_model('mobilenetv2_100', pretrained=False, features_only=True)
        # weights_path = '~/Guard-Net/mobilenetv2_100_ra-b33bc2c4.pth'
        # model.load_state_dict(torch.load(weights_path), strict=False)


        layers = [1,2,3,5,6]
        chans = [16, 24, 32, 96, 160]
        self.conv_stem = model.conv_stem
        self.bn1 = model.bn1
        # import pdb
        # pdb.set_trace()
        # self.act1 = model.act1  # 报错

        self.block0 = torch.nn.Sequential(*model.blocks[0:layers[0]])
        self.block1 = torch.nn.Sequential(*model.blocks[layers[0]:layers[1]])
        self.block2 = torch.nn.Sequential(*model.blocks[layers[1]:layers[2]])
        self.block3 = torch.nn.Sequential(*model.blocks[layers[2]:layers[3]])
        self.block4 = torch.nn.Sequential(*model.blocks[layers[3]:layers[4]])

        # self.deconv32_16 = Conv2x(chans[4], chans[3], deconv=True, concat=True)



    def forward(self, x):
        x = self.bn1(self.conv_stem(x))
        # x = self.act1(self.bn1(self.conv_stem(x)))
        x2 = self.block0(x)
        x4 = self.block1(x2)
        # return x4,x4,x4,x4
        x8 = self.block2(x4)
        x16 = self.block3(x8)
        x32 = self.block4(x16)
        return [x4, x8, x16, x32]

class FeatUp(SubModule):
    def __init__(self):
        super(FeatUp, self).__init__()
        chans = [16, 24, 32, 96, 160]
        self.deconv32_16 = Conv2x(chans[4], chans[3], deconv=True, concat=True)
        self.deconv16_8 = Conv2x(chans[3]*2, chans[2], deconv=True, concat=True)
        self.deconv8_4 = Conv2x(chans[2]*2, chans[1], deconv=True, concat=True)
        self.conv4 = BasicConv(chans[1]*2, chans[1]*2, kernel_size=3, stride=1, padding=1)

        self.weight_init()

    def forward(self, featL, featR=None):
        x4, x8, x16, x32 = featL

        y4, y8, y16, y32 = featR
        x16 = self.deconv32_16(x32, x16)
        y16 = self.deconv32_16(y32, y16)
        
        x8 = self.deconv16_8(x16, x8)
        y8 = self.deconv16_8(y16, y8)
        x4 = self.deconv8_4(x8, x4)
        y4 = self.deconv8_4(y8, y4)
        x4 = self.conv4(x4)
        y4 = self.conv4(y4)

        return [x4, x8, x16, x32], [y4, y8, y16, y32]

class channelAtt(SubModule):
    def __init__(self, cv_chan, im_chan):
        super(channelAtt, self).__init__()

        self.im_att = nn.Sequential(
            BasicConv(im_chan, im_chan//2, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(im_chan//2, cv_chan, 1))

        self.weight_init()

    def forward(self, cv, im):
        '''
        '''
        channel_att = self.im_att(im).unsqueeze(2)
        cv = torch.sigmoid(channel_att)*cv
        return cv


class hourglass_att(nn.Module):
    def __init__(self, in_channels):
        super(hourglass_att, self).__init__()

        self.conv1 = nn.Sequential(BasicConv(in_channels, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))
                                    
        self.conv2 = nn.Sequential(BasicConv(in_channels*2, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))                             

        self.conv3 = nn.Sequential(BasicConv(in_channels*4, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*6, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1)) 


        self.conv3_up = BasicConv(in_channels*6, in_channels*4, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv2_up = BasicConv(in_channels*4, in_channels*2, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv1_up = BasicConv(in_channels*2, 1, deconv=True, is_3d=True, bn=False,
                                  relu=False, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.agg_0 = nn.Sequential(BasicConv(in_channels*8, in_channels*4, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1),)

        self.agg_1 = nn.Sequential(BasicConv(in_channels*4, in_channels*2, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1),)


        self.feature_att_8 = channelAtt(in_channels*2, 64)
        self.feature_att_16 = channelAtt(in_channels*4, 192)
        self.feature_att_32 = channelAtt(in_channels*6, 160)
        self.feature_att_up_16 = channelAtt(in_channels*4, 192)
        self.feature_att_up_8 = channelAtt(in_channels*2, 64)

    def forward(self, x, imgs):
        conv1 = self.conv1(x)
        conv1 = self.feature_att_8(conv1, imgs[1])

        conv2 = self.conv2(conv1)
        conv2 = self.feature_att_16(conv2, imgs[2])

        conv3 = self.conv3(conv2)
        conv3 = self.feature_att_32(conv3, imgs[3])

        conv3_up = self.conv3_up(conv3)

        conv2 = torch.cat((conv3_up, conv2), dim=1)
        conv2 = self.agg_0(conv2)
        conv2 = self.feature_att_up_16(conv2, imgs[2])

        conv2_up = self.conv2_up(conv2)

        conv1 = torch.cat((conv2_up, conv1), dim=1)
        conv1 = self.agg_1(conv1)
        conv1 = self.feature_att_up_8(conv1, imgs[1])

        conv = self.conv1_up(conv1)

        return conv


class AxialConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

class CoEx(nn.Module):
    def __init__(self, maxdisp):
        super(CoEx, self).__init__()
        self.maxdisp = maxdisp 
        self.feature = Feature()
        self.feature_up = FeatUp()
        chans = [16, 24, 32, 96, 160]

        self.stem_2 = nn.Sequential(
            BasicConv(3, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU()
            )
        self.stem_4 = nn.Sequential(
            BasicConv(32, 48, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(48, 48, 3, 1, 1, bias=False),
            nn.BatchNorm2d(48), nn.ReLU()
            )

        self.spx = nn.Sequential(nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1),)
        self.spx_2 = Conv2x(24, 32, True)
        self.spx_4 = nn.Sequential(
            BasicConv(96, 24, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(24, 24, 3, 1, 1, bias=False),
            nn.BatchNorm2d(24), nn.ReLU()
            )

        self.conv = BasicConv(96, 48, kernel_size=3, padding=1, stride=1)
        self.desc = nn.Conv2d(48, 48, kernel_size=1, padding=0, stride=1)

        self.corr_feature_att_4 = channelAtt(8, 96)
        self.hourglass_att = hourglass_att(8)
        self.corr_stem = BasicConv(1, 8, is_3d=True, kernel_size=3, stride=1, padding=1)

        model_config = AxialConfig(embd_pdrop=0.0, resid_pdrop=0.0, n_embd=48, block_size=32,
                                     attn_pdrop=0.0, n_layer=8, n_head=8)
        self.blocks = []
        for _ in range(model_config.n_layer // 2):
            self.blocks.append(BlockAxial(model_config))
            # self.blocks.append(my_Block_2(model_config))
        self.blocks = nn.Sequential(*self.blocks)
        self.aspp = ASPP(48, [6, 12, 18])
        self.group = BasicConv(96, 48, kernel_size=3, padding=1, stride=1)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(2*torch.ones(1))

        self.uncertainty_refine = ContextAdjustmentLayer(8, 32, 4)

        

    def forward(self, left, right):
        # with torch.no_grad():

        features_left = self.feature(left)
        features_right = self.feature(right)

        features_left, features_right = self.feature_up(features_left, features_right)
        

        # torch.Size([48, 32, 128, 256])
        stem_2x = self.stem_2(left)
        # torch.Size([48, 48, 64, 128])
        stem_4x = self.stem_4(stem_2x)
        stem_2y = self.stem_2(right)
        stem_4y = self.stem_4(stem_2y)

        
        stem_4x_aspp = self.aspp(stem_4x)
        stem_4y_aspp = self.aspp(stem_4y)

        stem_4x_axial = self.blocks(stem_4x)
        stem_4y_axial = self.blocks(stem_4y)

        stem_4x = torch.cat((stem_4x_axial, stem_4x_aspp), 1)
        stem_4y = torch.cat((stem_4y_axial, stem_4y_aspp), 1)

        stem_4x = self.group(stem_4x)
        stem_4y = self.group(stem_4y)


        # torch.Size([48, 48+48, 64, 128])
        features_left[0] = torch.cat((features_left[0], stem_4x), 1)
        features_right[0] = torch.cat((features_right[0], stem_4y), 1)
        

        match_left = self.desc(self.conv(features_left[0]))
        match_right = self.desc(self.conv(features_right[0]))


        # torch.Size([48, 1, 48, 64, 128])
        corr_volume = build_norm_correlation_volume(match_left, match_right, self.maxdisp//4)
        
        # torch.Size([48, 8, 48, 64, 128])
        corr_volume = self.corr_stem(corr_volume)
        # torch.Size([48, 8, 48, 64, 128])
        cost = self.corr_feature_att_4(corr_volume, features_left[0]) 
        # torch.Size([48, 1, 48, 64, 128])
        cost = self.hourglass_att(cost, features_left) 

        # torch.Size([48, 24, 64, 128])
        xspx = self.spx_4(features_left[0])
        # torch.Size([48, 64, 128, 256])
        xspx = self.spx_2(xspx, stem_2x)
        # torch.Size([48, 9, 256, 512])
        spx_pred = self.spx(xspx)
        spx_pred = F.softmax(spx_pred, 1)
        
        # torch.Size([48])
        disp_samples = torch.arange(0, self.maxdisp//4, dtype=cost.dtype, device=cost.device)
        # torch.Size([40, 48, 64, 128])
        disp_samples = disp_samples.view(1, self.maxdisp//4, 1, 1).repeat(cost.shape[0],1,cost.shape[3],cost.shape[4])
        # torch.Size([40, 1, 64, 128])
        pred = regression_topk(cost.squeeze(1), disp_samples, 2)
        # torch.Size([40, 256, 512])
        pred_up = context_upsample(pred, spx_pred)


        P = F.interpolate(cost, size=(cost.shape[2]*4, cost.shape[3]*4, cost.shape[4]*4), mode='trilinear')
        P_init = F.softmax(P, 2)
        P_init = P_init.squeeze(1)

        D_init = pred_up*4

        U = disparity_variance(P_init, self.maxdisp, D_init.unsqueeze(1)) # torch.Size([40, 1, 64, 128])
        pred_variance = self.beta + self.gamma * U
        un_map_confidence = torch.sigmoid(pred_variance)

        pred_up = self.uncertainty_refine(D_init.unsqueeze(1), left, un_map_confidence)
        pred_up = pred_up.squeeze(1)


        if self.training:
            return [pred_up, pred.squeeze(1)*4]

        else:
            return [pred_up], [un_map_confidence]
