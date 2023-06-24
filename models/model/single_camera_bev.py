import numpy as np
import torch
import torchvision as tv
from torch import nn
from utils.homograph_util import homograph


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
        x = x.view(list(x.size()[:2]) + [self.image_featmap_size[1] * self.image_featmap_size[2], ])  # 这个 B,C,H*W x(32,512,576)
        bev_view = self.fc_transform(x)  # 拿出一个视角 bev_view(32,512,125)
        bev_view = bev_view.view(list(bev_view.size()[:2]) + [self.space_featmap_size[1], self.space_featmap_size[2]])  # bev_view(32,512,25,5)
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
class HG_CNN(nn.Module):
    def __init__(self):
        super(HG_CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.Conv2d(6, 12, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2)
            nn.MaxPool2d(kernel_size=4, stride=4)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(12, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(24, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2)
            nn.MaxPool2d(kernel_size=4, stride=4)
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
        # self.layer4 = nn.Sequential(
        #     nn.Conv2d(192, 128, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        #     nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2)
        # )
        # self.layer5 = nn.Sequential(
        #     nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        #     nn.Conv2d(128, 96, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(96),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2)
        # )
        self.layer6 = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(32 * 33 * 60, 258),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(258, 8)
        )
        hg_mlp_init_module(self.layer1)
        hg_mlp_init_module(self.layer2)
        hg_mlp_init_module(self.layer3)
        # hg_mlp_init_module(self.layer4)
        # hg_mlp_init_module(self.layer5)
        hg_mlp_init_module(self.layer6)

    def forward(self, x):
        # img (8,1080,1920,3) -> (8,3,1080,1920)
        # x = x.permute(0, 3, 1, 2).float()
        out = self.layer1(x)  # x img(8,3,1080,1920) -> out (8,12,540,960)
        out = self.layer2(out)  # out (8,12,540,960) -> out (8,48,270,480)
        out = self.layer3(out)  # out (8,48,270,480) -> out (8,192,135,240)
        # out = self.layer4(out)  # out (8,192,135,240) -> out (8,128,67,120)
        # out = self.layer5(out)  # out (8,128,67,120) -> out (8,96,33,60)
        out = self.layer6(out)  # out (8,96,33,60) -> out (8,32,33,60)
        out = out.contiguous().view(x.size(0), -1)
        out = self.fc(out)
        return out  # out(8,8)
class HG_MLP(nn.Module): # hg_mtx_feat (32,32,18,32) hg_mtx(32,9)
    def __init__(self, image_featmap_size, space_featmap_size):
        super(HG_MLP, self).__init__()
        ic, ih, iw = image_featmap_size  # s32transformer:(32, 18, 32)
        sc, sh, sw = space_featmap_size  # s32transformer:(1, 3, 3)
        self.image_featmap_size = image_featmap_size
        self.space_featmap_size = space_featmap_size
        self.fc_transform = nn.Sequential(
            nn.Linear(ih * iw, sh * sw),  # ih*iw=18x32=576 sh*sw=3x3=9
            nn.ReLU(),
            nn.Linear(sh * sw, sh * sw),  # sh*sw=3x3=9 sh*sw=3x3=9
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=ic, out_channels=sc, kernel_size=1 * 1, stride=1, bias=False),
            nn.BatchNorm2d(sc),
            nn.ReLU(), )
    def forward(self, x):  # x(32,32,18,32)
        x = x.view(list(x.size()[:2]) + [self.image_featmap_size[1] * self.image_featmap_size[2], ])  # B,C,H*W x(32,32,576)
        bev_view = self.fc_transform(x)  # x(32,32,576) bev_view(32,32,9)
        bev_view = bev_view.view(list(bev_view.size()[:2]) + [self.space_featmap_size[1], self.space_featmap_size[2]])  # bev_view(32,32,3,3)
        # bev_view = self.conv1(bev_view)
        bev_view = bev_view.mean(1)
        bev_view = bev_view.view(list(bev_view.size()[:1]) + [self.space_featmap_size[1]*self.space_featmap_size[2]])  # bev_view(32,9)
        return bev_view
# model
# ResNet34 骨干网络 (self.bb)，在 ImageNet 上进行预训练。
# 一个下采样层 (self.down)，用于减小特征图的空间维度。
# 两个全连接变换层 (self.s32transformer 和 self.s64transformer)，将 ResNet 骨干网络的特征图转换为 BEV 表示。
# 车道线检测头 (self.lane_head)，以 BEV 表示作为输入，输出表示检测到的车道线的张量。
# 可选的 2D 图像车道线检测头 (self.lane_head_2d)，以 ResNet 骨干网络的输出作为输入，输出表示原始图像中检测到的车道线的张量。
class BEV_LaneDet(nn.Module):  # BEV-LaneDet
    def __init__(self, bev_shape, output_2d_shape, train=True):
        super(BEV_LaneDet, self).__init__()

        ''' down pre '''
        self.down_pre = naive_init_module(
            nn.Sequential(
                Residual(
                    module=nn.Sequential(
                        nn.Conv2d(3, 6, kernel_size=3, stride=2, padding=1),  # S2
                        nn.BatchNorm2d(6),
                        nn.ReLU(),
                        nn.Conv2d(6, 6, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(6)
                    ),
                    downsample=nn.Conv2d(3, 6, kernel_size=3, stride=2, padding=1),
                ),
                Residual(
                    module=nn.Sequential(
                        nn.Conv2d(6, 12, kernel_size=3, stride=2, padding=1),  # S4
                        nn.BatchNorm2d(12),
                        nn.ReLU(),
                        nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(12)
                    ),
                    downsample=nn.Conv2d(6, 12, kernel_size=3, stride=2, padding=1),
                ),
                Residual(
                    module=nn.Sequential(
                        nn.Conv2d(12, 24, kernel_size=3, stride=2, padding=1),  # S8
                        nn.BatchNorm2d(24),
                        nn.ReLU(),
                        nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(24)
                    ),
                    downsample=nn.Conv2d(12, 24, kernel_size=3, stride=2, padding=1),
                ),
                Residual(
                    module=nn.Sequential(
                        nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1),  # S16
                        nn.BatchNorm2d(48),
                        nn.ReLU(),
                        nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(48)
                    ),
                    downsample=nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1),
                ),
                Residual(
                    module=nn.Sequential(
                        nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1),  # S32
                        nn.BatchNorm2d(96),
                        nn.ReLU(),
                        nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(96)
                    ),
                    downsample=nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1),
                ),
                # Residual(
                #     module=nn.Sequential(
                #         nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1),  # S64
                #         nn.BatchNorm2d(128),
                #         nn.ReLU(),
                #         nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                #         nn.BatchNorm2d(128)
                #     ),
                #     downsample=nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1),
                # ),
                nn.Conv2d(96, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU()
            ))

        self.head = InstanceEmbedding(32, 2)
        naive_init_module(self.head)
        # self.bb_pre = nn.Sequential(*list(tv.models.resnet18(pretrained=True).children())[:-2])
        ''' backbone '''
        self.bb = nn.Sequential(*list(tv.models.resnet18(pretrained=True).children())[:-2])
        ''' HomograpNet '''

        # self.hg = nn.Sequential( #
        #     # nn.Flatten(),
        #     nn.Linear(in_features=32 * 18 * 32, out_features=512, bias=True),
        #     nn.ReLU(),
        #     nn.Linear(in_features=512, out_features=256, bias=True),
        #     nn.ReLU(),
        #     nn.Linear(in_features=256, out_features=256, bias=True),
        #     nn.ReLU(),
        #     nn.Linear(in_features=256, out_features=9, bias=True)
        # )

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
        # self.hg = HG_MLP((32, 18, 32), (1, 3, 3))  # hg_mtx_feat (32,32,18,32) hg_mtx(32,9)
        self.hg = HG_CNN()  # hg_mtx_feat (32,32,18,32) hg_mtx(32,9)
        self.s32transformer = FCTransform_((512, 18, 32), (256, 25, 5))
        self.s64transformer = FCTransform_((1024, 9, 16), (256, 25, 5))
        self.lane_head = LaneHeadResidual_Instance_with_offset_z(bev_shape, input_channel=512)
        self.is_train = train
        if self.is_train:
            self.lane_head_2d = LaneHeadResidual_Instance(output_2d_shape, input_channel=512)

    def forward(self, img, img_gt=None, configs=None):  # img (32,3,576,1024)  img_gt (32,1080,1920)
        # hg_mtx_feat = self.down_pre(img)  # img (32,3,576,1024) hg_mtx_feat (32,32,18,32)
        # hg_mtx = self.hg(hg_mtx_feat)  # hg_mtx_feat (32,32,18,32) hg_mtx(32,9)
        hg_mtx = self.hg(img)  # img(8,1080,1920,3)
        print(np.array(hg_mtx[0]))
        img_vt, image_gt_instance, image_gt_segment = homograph(img, img_gt, hg_mtx, configs)
        img_vt_s32 = self.bb(img_vt)  # img_vt (32,3,576,1024) img_vt_s32 (32,512,18,32)
        img_vt_s64 = self.down(img_vt_s32)  # img_vt_s32 (32,512,18,32) img_s64 (32,1024,9,16)
        bev_32 = self.s32transformer(img_vt_s32)  # img_vt_s32(32,512,18,32) bev_32 (32,256,25,5)
        bev_64 = self.s64transformer(img_vt_s64)  # img_s64 (32,1024,9,16) bev_64 (32,256,25,5)
        bev = torch.cat([bev_64, bev_32], dim=1)  # bev (8,512,25,5)
        if self.is_train:
            return image_gt_instance, image_gt_segment, self.lane_head(bev), self.lane_head_2d(img_vt_s32)
        else:
            return image_gt_instance, image_gt_segment, self.lane_head(bev)
