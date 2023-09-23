import numpy as np
import torch
import torchvision as tv
from torch import nn
from utils.homograph_util import homograph
import torch.nn.functional as F
import kornia.geometry.transform as kgt
import kornia.geometry.conversions as kgc


def naive_init_module(mod):
    for m in mod.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    return mod


def hg_mlp_init_module(mod):
    for m in mod.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    return mod


class InstanceEmbedding_offset_y_z(nn.Module):
    def __init__(self, ci, co=1):
        super(InstanceEmbedding_offset_y_z, self).__init__()
        self.neck_new = nn.Sequential(
            # SELayer(ci),
            nn.Conv2d(ci, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, ci, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ci),
            nn.ReLU(),
        )

        self.ms_new = nn.Sequential(
            # nn.Dropout2d(0.2),
            nn.Conv2d(ci, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 1, 3, 1, 1, bias=True)
        )

        self.m_offset_new = nn.Sequential(
            # nn.Dropout2d(0.2),
            nn.Conv2d(ci, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 1, 3, 1, 1, bias=True)
        )

        self.m_z = nn.Sequential(
            # nn.Dropout2d(0.2),
            nn.Conv2d(ci, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 1, 3, 1, 1, bias=True)
        )

        self.me_new = nn.Sequential(
            # nn.Dropout2d(0.2),
            nn.Conv2d(ci, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, co, 3, 1, 1, bias=True)
        )

        naive_init_module(self.ms_new)
        naive_init_module(self.me_new)
        naive_init_module(self.m_offset_new)
        naive_init_module(self.m_z)
        naive_init_module(self.neck_new)

    def forward(self, x):
        feat = self.neck_new(x)
        return self.ms_new(feat), self.me_new(feat), self.m_offset_new(feat), self.m_z(feat)


class InstanceEmbedding(nn.Module):
    def __init__(self, ci, co=1):
        super(InstanceEmbedding, self).__init__()
        self.neck = nn.Sequential(
            # SELayer(ci),
            nn.Conv2d(ci, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, ci, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ci),
            nn.ReLU(),
        )

        self.ms = nn.Sequential(
            # nn.Dropout2d(0.2),
            nn.Conv2d(ci, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 1, 3, 1, 1, bias=True)
        )

        self.me = nn.Sequential(
            # nn.Dropout2d(0.2),
            nn.Conv2d(ci, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, co, 3, 1, 1, bias=True)
        )

        naive_init_module(self.ms)
        naive_init_module(self.me)
        naive_init_module(self.neck)

    def forward(self, x):
        feat = self.neck(x)
        return self.ms(feat), self.me(feat)


class LaneHeadResidual_Instance_with_offset_z(nn.Module):
    def __init__(self, output_size, input_channel=256):
        super(LaneHeadResidual_Instance_with_offset_z, self).__init__()

        self.bev_up_new = nn.Sequential(
            nn.Upsample(scale_factor=2),  # 
            Residual(
                module=nn.Sequential(
                    nn.Conv2d(input_channel, 64, 3, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Dropout2d(p=0.2),
                    nn.Conv2d(64, 128, 3, padding=1, bias=False),
                    nn.BatchNorm2d(128),
                ),
                downsample=nn.Conv2d(input_channel, 128, 1),
            ),
            nn.Upsample(size=output_size),  #
            Residual(
                module=nn.Sequential(
                    nn.Conv2d(128, 64, 3, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Dropout2d(p=0.2),
                    nn.Conv2d(64, 64, 3, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                    # nn.ReLU(),
                ),
                downsample=nn.Conv2d(128, 64, 1),
            ),
        )
        self.head = InstanceEmbedding_offset_y_z(64, 2)
        naive_init_module(self.head)
        naive_init_module(self.bev_up_new)

    def forward(self, bev_x):
        bev_feat = self.bev_up_new(bev_x)
        return self.head(bev_feat)


class LaneHeadResidual_Instance(nn.Module):
    def __init__(self, output_size, input_channel=256):
        super(LaneHeadResidual_Instance, self).__init__()

        self.bev_up = nn.Sequential(
            nn.Upsample(scale_factor=2),  # 60x 24
            Residual(
                module=nn.Sequential(
                    nn.Conv2d(input_channel, 64, 3, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Dropout2d(p=0.2),
                    nn.Conv2d(64, 128, 3, padding=1, bias=False),
                    nn.BatchNorm2d(128),
                ),
                downsample=nn.Conv2d(input_channel, 128, 1),
            ),
            nn.Upsample(scale_factor=2),  # 120 x 48
            Residual(
                module=nn.Sequential(
                    nn.Conv2d(128, 64, 3, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Dropout2d(p=0.2),
                    nn.Conv2d(64, 32, 3, padding=1, bias=False),
                    nn.BatchNorm2d(32),
                    # nn.ReLU(),
                ),
                downsample=nn.Conv2d(128, 32, 1),
            ),

            nn.Upsample(size=output_size),  # 300 x 120
            Residual(
                module=nn.Sequential(
                    nn.Conv2d(32, 16, 3, padding=1, bias=False),
                    nn.BatchNorm2d(16),
                    nn.ReLU(),
                    nn.Dropout2d(p=0.2),
                    nn.Conv2d(16, 32, 3, padding=1, bias=False),
                    nn.BatchNorm2d(32),
                )
            ),
        )

        self.head = InstanceEmbedding(32, 2)
        naive_init_module(self.head)
        naive_init_module(self.bev_up)

    def forward(self, bev_x):
        bev_feat = self.bev_up(bev_x)
        return self.head(bev_feat)


class FCTransform_(nn.Module):
    def __init__(self, image_featmap_size, space_featmap_size):
        super(FCTransform_, self).__init__()
        ic, ih, iw = image_featmap_size  # (256, 16, 16)  s32transformer:(512, 18, 32)
        sc, sh, sw = space_featmap_size  # (128, 16, 32)  s32transformer:(256, 25, 5)
        self.image_featmap_size = image_featmap_size
        self.space_featmap_size = space_featmap_size
        self.fc_transform = nn.Sequential(
            nn.Linear(ih * iw, sh * sw),  # ih*iw=18x32=576 sh*sw=25x5=125
            nn.ReLU(),
            nn.Linear(sh * sw, sh * sw),  # sh*sw=25x5=125 sh*sw=25x5=125
            nn.ReLU()
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=ic, out_channels=sc, kernel_size=1 * 1, stride=1, bias=False),
            nn.BatchNorm2d(sc),
            nn.ReLU(), )
        self.residual = Residual(
            module=nn.Sequential(
                nn.Conv2d(in_channels=sc, out_channels=sc, kernel_size=3, padding=1, stride=1, bias=False),
                nn.BatchNorm2d(sc),
            ))

    def forward(self, x):  # x(32,512,18,32)
        x = x.view(list(x.size()[:2]) + [
            self.image_featmap_size[1] * self.image_featmap_size[2], ])  # 这个 B,C,H*W x(32,512,576)
        bev_view = self.fc_transform(x)  # 拿出一个视角 bev_view(32,512,125)
        bev_view = bev_view.view(list(bev_view.size()[:2]) + [self.space_featmap_size[1],
                                                              self.space_featmap_size[2]])  # bev_view(32,512,25,5)
        bev_view = self.conv1(bev_view)  # bev_view (32,512,25,5)->(32,256,25,5)
        bev_view = self.residual(bev_view)  # bev_view (32,256,25,5)->(32,256,25,5)
        return bev_view


class Residual(nn.Module):
    def __init__(self, module, downsample=None):
        super(Residual, self).__init__()
        self.module = module
        self.downsample = downsample
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.module(x)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)


class HG_MLP(nn.Module):
    def __init__(self):
        super(HG_MLP, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.Conv2d(6, 12, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(12, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(24, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(48, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Conv2d(96, 192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(96, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(32 * 18 * 32, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 2),
            # nn.Tanh()
        )
        hg_mlp_init_module(self.layer1)
        hg_mlp_init_module(self.layer2)
        hg_mlp_init_module(self.layer3)
        hg_mlp_init_module(self.layer4)
        hg_mlp_init_module(self.layer5)
        hg_mlp_init_module(self.layer6)

    def forward(self, images, configs):
        out = self.layer1(images)  # x img(8,3,576,1024) -> out (8,12,288,512)
        out = self.layer2(out)  # out (8,12,288,512) -> out (8,48,144,256)
        out = self.layer3(out)  # out (8,48,144,256) -> out (8,192,72,128)
        out = self.layer4(out)  # out (8,192,72,128) -> out (8,128,36,64)
        out = self.layer5(out)  # out (8,128,36,64) -> out (8,96,18,32)
        out = self.layer6(out)  # out (8,96,18,32) -> out (8,32,18,32)
        out = out.contiguous().view(images.size(0), -1)
        out = self.fc(out)  # out(8,8)
        # img_warped, imgs_gt_inst, imgs_gt_seg, hg_mtxs = self.hg_transform(images, images_gt, out, configs)
        # [1.00068778e+00, 2.23062386e-20, 6.60278320e-01, 3.67842412e-16, 1.00068795e+00, 4.26676470e+01,
        #  4.27762242e-19, 0.00000000e+00]
        tmp = torch.zeros((images.shape[0], 9), requires_grad=True).cuda()
        H_indices = torch.tensor([[2, 5]]).expand(images.shape[0], 2).cuda()
        tmp.scatter_(dim=1, index=H_indices, src=out)
        trans_matrix_gt = torch.tensor([[1.0, 0.0, 1.0, 0.0, 1.0, 10.0, 0.0, 0.0, 1.0]], requires_grad=True).cuda()
        # out = torch.zeros((8, 8),dtype=torch.float32).cuda()
        h_matrix = tmp + trans_matrix_gt
        img_warped, hg_mtxs = self.hg_transform(images, h_matrix, configs)
        return img_warped, hg_mtxs

    def hg_transform(self, images, hg_mtxs, configs):
        # hg_mtx (8,8) -> (8,3,3)
        # hg_mtxs = torch.cat((hg_mtxs, torch.ones(hg_mtxs.shape[0], 1).cuda()), dim=1)
        hg_mtxs = hg_mtxs.view((hg_mtxs.shape[0], 3, 3))  # hg_mtxs(16,3,3)

        # images(16,3,576,1024) img_s32 (16,512,18,32) images_gt (16,1280,1920) hg_mtxs(16,9)
        batch_size = images.shape[0]
        output_2d_h, output_2d_w = configs.output_2d_shape[0], configs.output_2d_shape[1]
        # input_shape 数据增强的resize尺寸，也是源代码中进入backbone的尺寸。
        # 处理完homograph之后，图像变换到该尺寸作为后序backbone的输入
        img_s32_h, img_s32_w, img_s32_c = configs.input_shape[0], configs.input_shape[1], images.shape[1]
        img_vt_s32_hg_shape = (img_s32_h, img_s32_w)  # (1024,576)

        images.requires_grad_()

        # hg_mtxs_image = kgc.denormalize_homography(hg_mtxs, (img_s32_h, img_s32_w), (img_s32_h, img_s32_w))
        images_warped = kgt.warp_perspective(images, hg_mtxs, img_vt_s32_hg_shape)

        return images_warped.float(), hg_mtxs.float()


# model
# ResNet34 骨干网络 (self.bb)，在 ImageNet 上进行预训练。
# 一个下采样层 (self.down)，用于减小特征图的空间维度。
# 两个全连接变换层 (self.s32transformer 和 self.s64transformer)，将 ResNet 骨干网络的特征图转换为 BEV 表示。
# 车道线检测头 (self.lane_head)，以 BEV 表示作为输入，输出表示检测到的车道线的张量。
# 可选的 2D 图像车道线检测头 (self.lane_head_2d)，以 ResNet 骨干网络的输出作为输入，输出表示原始图像中检测到的车道线的张量。
class BEV_LaneDet(nn.Module):  # BEV-LaneDet
    def __init__(self, bev_shape, output_2d_shape, train=True):
        super(BEV_LaneDet, self).__init__()
        self.head = InstanceEmbedding(32, 2)
        naive_init_module(self.head)
        ''' backbone '''
        self.bb = nn.Sequential(*list(tv.models.resnet34(pretrained=True).children())[:-2])
        self.down = naive_init_module(
            Residual(
                module=nn.Sequential(
                    nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),  # S64
                    nn.BatchNorm2d(1024),
                    nn.ReLU(),
                    nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(1024)

                ),
                downsample=nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            )
        )
        self.hg = HG_MLP()  # hg_mtx_feat (32,32,18,32) hg_mtx(32,9)
        self.hg_util = homograph
        self.s32transformer = FCTransform_((512, 18, 32), (256, 25, 5))
        self.s64transformer = FCTransform_((1024, 9, 16), (256, 25, 5))
        self.lane_head = LaneHeadResidual_Instance_with_offset_z(bev_shape, input_channel=512)
        self.is_train = train
        if self.is_train:
            self.lane_head_2d = LaneHeadResidual_Instance(output_2d_shape, input_channel=512)

    def forward(self, img, configs=None):  # img (32,3,576,1024)  img_gt (32,1080,1920)

        img_vt, hg_mtxs = self.hg(img, configs)  # img(8,1080,1920,3) hg_mtx(8,8)

        # hg_mtxs = self.hg(img)  # img(8,1080,1920,3) hg_mtx(8,8)
        # # hg_mtx (8,8) -> (8,3,3)
        # hg_mtxs = F.normalize(hg_mtxs, dim=1, p=2, eps=1e-6)  # H^-1
        # hg_mtxs = torch.cat((hg_mtxs, torch.ones(hg_mtxs.shape[0], 1).cuda()), dim=1)
        # hg_mtxs = hg_mtxs.view((hg_mtxs.shape[0], 3, 3))  # hg_mtxs(16,3,3)
        #
        # img_vt, imgs_gt_inst, imgs_gt_seg = self.hg_util(img, img_gt, hg_mtxs, configs)

        img_vt_s32 = self.bb(img_vt)  # img_vt (32,3,576,1024) img_vt_s32 (32,512,18,32)
        img_vt_s64 = self.down(img_vt_s32)  # img_vt_s32 (32,512,18,32) img_s64 (32,1024,9,16)
        bev_32 = self.s32transformer(img_vt_s32)  # img_vt_s32(32,512,18,32) bev_32 (32,256,25,5)
        bev_64 = self.s64transformer(img_vt_s64)  # img_s64 (32,1024,9,16) bev_64 (32,256,25,5)
        bev = torch.cat([bev_64, bev_32], dim=1)  # bev (8,512,25,5)
        if self.is_train:
            return img_vt, hg_mtxs, self.lane_head(bev), self.lane_head_2d(img_vt_s32)
        else:
            return img_vt, hg_mtxs, self.lane_head(bev)
