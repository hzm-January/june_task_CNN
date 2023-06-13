import cv2
import torch
import numpy as np


def homograph(images, images_gt, homograph_matrixes, configs):
    homograph_matrixes = homograph_matrixes.view((images.shape[0], 3, 3))
    images_gt_instance = torch.zeros(images_gt.shape[0],1, configs.output_2d_shape[0], configs.output_2d_shape[1]).cuda()
    images_gt_segment = torch.zeros(images_gt.shape[0],1,configs.output_2d_shape[0], configs.output_2d_shape[1]).cuda()
    images_res = torch.zeros(images.shape[0],images.shape[3],configs.resize_shape[0],configs.resize_shape[1]).cuda()
    for i in range(images.shape[0]):
        image = images[i]
        image_gt = images_gt[i]
        homograph_matrix = homograph_matrixes[i] / homograph_matrixes[i][-1][-1]
        image = cv2.warpPerspective(image.detach().cpu().numpy(), homograph_matrix.detach().cpu().numpy(), configs.vc_config['vc_image_shape'])
        image_gt = cv2.warpPerspective(image_gt.detach().cpu().numpy(), homograph_matrix.detach().cpu().numpy(), configs.vc_config['vc_image_shape'])

        transformed = configs.train_trans(image=image)  # 数据增强
        image = transformed["image"]
        ''' 2d gt '''
        image_gt = cv2.resize(image_gt, (configs.output_2d_shape[1], configs.output_2d_shape[0]),
                              interpolation=cv2.INTER_NEAREST)
        image = torch.tensor(image, dtype=torch.float).cuda()
        image_gt = torch.tensor(image_gt, dtype=torch.float).cuda()
        image_gt_instance = torch.tensor(image_gt).unsqueeze(0)  # h, w, c
        image_gt_segment = torch.clone(image_gt_instance)
        image_gt_segment[image_gt_segment > 0] = 1
        images_res[i] = image
        # images_gt[i] = image_gt
        images_gt_instance[i] = image_gt_instance
        images_gt_segment[i] = image_gt_segment
    return images_res, images_gt_instance.float(), images_gt_segment.float()
