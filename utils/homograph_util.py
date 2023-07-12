import cv2
import torch
import numpy as np
import torchgeometry as tgm
import kornia.geometry.transform as kgt
import kornia.enhance as keh
import kornia.geometry.conversions as kgc
import torch.nn.functional as F
def homograph(images, images_gt, hg_mtxs, configs):
    # images(16,3,576,1024) img_s32 (16,512,18,32) images_gt (16,1280,1920) hg_mtxs(16,9)
    batch_size = images.shape[0]
    output_2d_h, output_2d_w = configs.output_2d_shape[0], configs.output_2d_shape[1]
    # input_shape 数据增强的resize尺寸，也是源代码中进入backbone的尺寸。
    # 处理完homograph之后，图像变换到该尺寸作为后序backbone的输入
    img_s32_h, img_s32_w, img_s32_c = configs.input_shape[0], configs.input_shape[1], images.shape[1]
    img_vt_s32_hg_shape = (img_s32_h, img_s32_w)  # (1024,576)
    image_gt_hg_shape = configs.vc_config['vc_image_shape']  # (1920, 1280)
    images_gt_instance = torch.zeros(batch_size, 1, output_2d_h, output_2d_w).cuda()  # (16,1,144,256)
    images_gt_segment = torch.zeros(batch_size, 1, output_2d_h, output_2d_w).cuda()  # (16,1,144,256)
    # img_s32 (8,512,18,32)
    # imgs_vt_s32 = torch.zeros_like(images)  # (16,3,576,1024)
    # images = images.permute(0,2, 3, 1)  # (bz,576,1024,3)
    # image = cv2.warpPerspective(image.clone().cpu().numpy(), hg_mtx.clone().cpu().numpy(), img_vt_s32_hg_shape)
    # hg_mtxs = keh.normalize(hg_mtxs.unsqueeze(0), mean=hg_mtxs.mean(dim=(1, 2)), std=hg_mtxs.std(dim=(1, 2))).squeeze(0)
    # hg_mtxs = hg_mtxs-mean/std
    hg_mtxs = F.normalize(hg_mtxs, dim=(1, 2), p=2, eps=1e-6)  # H^-1
    hg_mtxs_image = kgc.denormalize_homography(hg_mtxs, (img_s32_h, img_s32_w), (img_s32_h, img_s32_w))
    images_warped = kgt.warp_perspective(images.clone(), hg_mtxs_image, img_vt_s32_hg_shape)
    hg_mtxs_image_gt = kgc.denormalize_homography(hg_mtxs, configs.output_2d_shape, configs.output_2d_shape)

    # images = images.permute(0, 3, 1, 2)
    if images_gt is not None:
        images_gt_warped = images_gt.clone().unsqueeze(1)
        images_gt_warped = kgt.warp_perspective(images_gt_warped, hg_mtxs_image_gt, configs.output_2d_shape)
        # images_gt = images_gt.squeeze(1)
        images_gt_warped = torch.round(images_gt_warped)
        for i in range(batch_size):
            image_gt = images_gt_warped[i]
            ''' 2d gt '''
            # image_gt = cv2.resize(image_gt, (output_2d_w, output_2d_h),  # (144,256)
            #                       interpolation=cv2.INTER_NEAREST)
            # image_gt = torch.tensor(image_gt, dtype=torch.float).cuda()
            # image_gt_instance = torch.tensor(image_gt).unsqueeze(0)  # (1, 144,256)
            image_gt_instance = torch.clone(image_gt)
            image_gt_segment = torch.clone(image_gt_instance)  # (1, 144,256)
            image_gt_segment[image_gt_segment > 0] = 1  # (1, 144,256)
            images_gt_instance[i], images_gt_segment[i] = image_gt_instance, image_gt_segment

    return images_warped.float(), images_gt_instance.float(), images_gt_segment.float()
