import torch
from torch import nn, Tensor
# from torch.nn.utils import weight_norm


class ResBlock(nn.Module):
    def __init__(self, n_feats: int, expansion_ratio: int, res_scale: int = 1.0):
        super(ResBlock, self).__init__()
        self.res_scale = res_scale
        self.module = nn.Sequential(
            (nn.Conv2d(n_feats + 1, n_feats * expansion_ratio, kernel_size=3, padding=1)),
            nn.ReLU(inplace=True),
            (nn.Conv2d(n_feats * expansion_ratio, n_feats, kernel_size=3, padding=1))
        )
        # 原来多一个weight_norm就会报错
        # self.module = nn.Sequential(
        #     weight_norm(nn.Conv2d(n_feats + 1, n_feats * expansion_ratio, kernel_size=3, padding=1)),
        #     nn.ReLU(inplace=True),
        #     weight_norm(nn.Conv2d(n_feats * expansion_ratio, n_feats, kernel_size=3, padding=1))
        # )

    def forward(self, x: torch.Tensor, disp: torch.Tensor):
        return x + self.module(torch.cat([disp, x], dim=1)) * self.res_scale


class ContextAdjustmentLayer(nn.Module):
    def __init__(self, num_blocks=8, feature_dim=16, expansion=3):
        super(ContextAdjustmentLayer, self).__init__()
        self.num_blocks = num_blocks

        # disp head
        self.in_conv = nn.Conv2d(4, feature_dim, kernel_size=3, padding=1)
        self.layers = nn.ModuleList([ResBlock(feature_dim, expansion) for _ in range(num_blocks)])
        self.out_conv = nn.Conv2d(feature_dim, 1, kernel_size=3, padding=1)
        

    def forward(self, disp_raw: Tensor, img: Tensor, un_map: Tensor):
        """
        :param disp_raw: raw disparity, [N,H,W]
        :param img: input left image, [N,3,H,W]
        :param un_map: un_map, [N,H,W]
        :return:
            disp_final: final disparity [N,1,H,W]
        """""
        # disp_raw = disp_raw.unsqueeze(1)
        # un_map = un_map.unsqueeze(1)
        # feat一直都是torch.Size([26, 16, 288, 576])
        feat = self.in_conv(torch.cat([disp_raw, img], dim=1))
        for layer in self.layers:
            feat = feat * un_map
            feat = layer(feat, disp_raw)
        disp_res = self.out_conv(feat)
        disp_final = disp_raw + disp_res

        return disp_final

if __name__ == '__main__':
    uncertainty_refine = ContextAdjustmentLayer(12, 32, 4)
    total = sum([param.nelement() for param in uncertainty_refine.parameters()])
    print("Number of parameter: %.2f Millions" % (total/1e6))
    # print(uncertainty_refine)