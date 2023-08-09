import torch
import kornia.geometry.transform as kgt
import kornia.geometry.conversions as kgc

def homograph(images, images_gt, hg_mtxs, configs):
    # images(16,3,576,1024) img_s32 (16,512,18,32) images_gt (16,1280,1920) hg_mtxs(16,9)
    batch_size = images.shape[0]
    output_2d_h, output_2d_w = configs.output_2d_shape[0], configs.output_2d_shape[1]
    # input_shape 数据增强的resize尺寸，也是源代码中进入backbone的尺寸。
    # 处理完homograph之后，图像变换到该尺寸作为后序backbone的输入
    img_s32_h, img_s32_w, img_s32_c = configs.input_shape[0], configs.input_shape[1], images.shape[1]
    img_vt_s32_hg_shape = (img_s32_h, img_s32_w)  # (1024,576)

    images_gt_instance = torch.zeros(batch_size, 1, output_2d_h, output_2d_w).cuda()  # (16,1,144,256)
    images_gt_segment = torch.zeros(batch_size, 1, output_2d_h, output_2d_w).cuda()  # (16,1,144,256)

    hg_mtxs_image = kgc.denormalize_homography(hg_mtxs, (img_s32_h, img_s32_w), (img_s32_h, img_s32_w))
    images_warped = kgt.warp_perspective(images, hg_mtxs_image, img_vt_s32_hg_shape)
    hg_mtxs_image_gt = kgc.denormalize_homography(hg_mtxs, configs.output_2d_shape, configs.output_2d_shape)

    # images = images.permute(0, 3, 1, 2)
    if images_gt is not None:
        images_gt_warped = images_gt.unsqueeze(1)
        images_gt_warped = kgt.warp_perspective(images_gt_warped, hg_mtxs_image_gt, configs.output_2d_shape)
        images_gt_warped = torch.round(images_gt_warped) #TODO 这里取round会不会吞掉梯度回传
        # images_gt_warped = ste_round(images_gt_warped) #TODO 这里取round会不会吞掉梯度回传

        for i in range(batch_size):
            image_gt = images_gt_warped[i]
            ''' 2d gt '''
            image_gt_instance = torch.clone(image_gt)
            image_gt_segment = torch.clone(image_gt_instance)  # (1, 144,256)
            image_gt_segment[image_gt_segment > 0] = 1  # (1, 144,256)
            images_gt_instance[i], images_gt_segment[i] = image_gt_instance, image_gt_segment

    return images_warped.float(), images_gt_instance.float(), images_gt_segment.float()

def ste_round(x):
    return torch.round(x) - x.detach() + x
