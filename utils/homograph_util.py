import cv2
import torch
import numpy as np
import torchgeometry as tgm

def homograph(images, images_gt, hg_mtxs, configs):
    # images(16,3,576,1024) img_s32 (16,512,18,32) images_gt (16,1280,1920) hg_mtxs(16,9)
    batch_size = images.shape[0]
    output_2d_w, output_2d_h = configs.output_2d_shape[0], configs.output_2d_shape[1]
    # input_shape 数据增强的resize尺寸，也是源代码中进入backbone的尺寸。
    # 处理完homograph之后，图像变换到该尺寸作为后序backbone的输入
    img_s32_w, img_s32_h, img_s32_c = configs.input_shape[0], configs.input_shape[1], images.shape[1]
    img_vt_s32_hg_shape = (img_s32_w, img_s32_h)  # (1024,576)
    image_gt_hg_shape = configs.vc_config['vc_image_shape']  # (1920, 1280)
    images_gt_instance = torch.zeros(batch_size, 1, output_2d_w, output_2d_h).cuda()  # (16,1,144,256)
    images_gt_segment = torch.zeros(batch_size, 1, output_2d_w, output_2d_h).cuda()  # (16,1,144,256)
    # img_s32 (8,512,18,32)
    # imgs_vt_s32 = torch.zeros_like(images)  # (16,3,576,1024)
    # images = images.permute(0,2, 3, 1)  # (bz,576,1024,3)
    # image = cv2.warpPerspective(image.clone().cpu().numpy(), hg_mtx.clone().cpu().numpy(), img_vt_s32_hg_shape)
    images = tgm.homography_warp(images, hg_mtxs, img_vt_s32_hg_shape, padding_mode="zeros")
    # images = images.permute(0, 3, 1, 2)
    if images_gt is not None:
        images_gt = images_gt.unsqueeze(1)
        images_gt = tgm.homography_warp(images_gt, hg_mtxs, configs.output_2d_shape, padding_mode="zeros")
        # images_gt = images_gt.squeeze(1)
        for i in range(batch_size):
            image_gt = images_gt[i]
            ''' 2d gt '''
            # image_gt = cv2.resize(image_gt, (output_2d_h, output_2d_w),  # (144,256)
            #                       interpolation=cv2.INTER_NEAREST)
            # image_gt = torch.tensor(image_gt, dtype=torch.float).cuda()
            # image_gt_instance = torch.tensor(image_gt).unsqueeze(0)  # (1, 144,256)
            image_gt_instance = torch.clone(image_gt)
            image_gt_segment = torch.clone(image_gt_instance)  # (1, 144,256)
            image_gt_segment[image_gt_segment > 0] = 1  # (1, 144,256)
            images_gt_instance[i], images_gt_segment[i] = image_gt_instance, image_gt_segment


    # for i in range(batch_size):
    #     image = images[i]  # images (3,576,1024) images_gt (1280,1920)
    #
    #     # image(1080,1920,3) trans_matrix(3,3) vc_image_shape(1920,1080) -> image(1080,1920,3)
    #     # image = cv2.warpPerspective(image, trans_matrix, self.vc_image_shape)
    #     # image_gt(1080,1920) trans_matrix(3,3) vc_image_shape(1920,1080) -> image_gt(1080,1920)
    #     # image_gt = cv2.warpPerspective(image_gt, trans_matrix, self.vc_image_shape)
    #
    #     hg_mtx = hg_mtxs[i]
    #     image = image.permute(1, 2, 0)  # (576,1024,3)
    #     # image = cv2.warpPerspective(image.clone().cpu().numpy(), hg_mtx.clone().cpu().numpy(), img_vt_s32_hg_shape)
    #     image = tgm.homography_warp(image, hg_mtx, img_vt_s32_hg_shape, padding_mode="zeros")
    #     image = torch.tensor(image, dtype=torch.float).permute(2, 0, 1).cuda()  # images (3,576,1024)
    #
    #     image_gt_instance = torch.zeros(configs.output_2d_shape)
    #     image_gt_segment = torch.zeros(configs.output_2d_shape)
    #     # if images_gt.numel() > 0:
    #     if images_gt is not None:
    #         image_gt = images_gt[i]
    #         # image_gt = cv2.warpPerspective(image_gt.clone().cpu().numpy(), hg_mtx.clone().cpu().numpy(),
    #         #                                image_gt_hg_shape)
    #         image_gt = tgm.homography_warp(image_gt, hg_mtx, image_gt_hg_shape, padding_mode="zeros")
    #         ''' 2d gt '''
    #         image_gt = cv2.resize(image_gt, (output_2d_h, output_2d_w),  # (144,256)
    #                               interpolation=cv2.INTER_NEAREST)
    #         image_gt = torch.tensor(image_gt, dtype=torch.float).cuda()
    #         image_gt_instance = torch.tensor(image_gt).unsqueeze(0)  # (1, 144,256)
    #         image_gt_segment = torch.clone(image_gt_instance)   # (1, 144,256)
    #         image_gt_segment[image_gt_segment > 0] = 1   # (1, 144,256)

        # imgs_vt_s32[i], images_gt_instance[i], images_gt_segment[i] = image, image_gt_instance, image_gt_segment
        # images_gt[i] = image_gt


    return images.float(), images_gt_instance.float(), images_gt_segment.float()
