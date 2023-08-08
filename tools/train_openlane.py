import sys
sys.path.append('/home/houzm/houzm/02_code/bev_lane_det-cnn')
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import torch.nn as nn
from models.util.load_model import load_checkpoint, resume_training
from models.util.save_model import save_model_dp
from models.loss import IoULoss, NDPushPullLoss
from utils.config_util import load_config_module
from sklearn.metrics import f1_score
import numpy as np
import os
import kornia.geometry.transform as kgt
import kornia.geometry.conversions as kgc
import kornia.enhance as keh
import torch.nn.functional as F
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

class Combine_Model_and_Loss(torch.nn.Module):
    def __init__(self, model):
        super(Combine_Model_and_Loss, self).__init__()
        self.model = model
        self.bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]))
        self.iou_loss = IoULoss()
        self.poopoo = NDPushPullLoss(1.0, 1., 1.0, 5.0, 200)
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        # self.sigmoid = nn.Sigmoid()

    def forward(self, inputs, images_gt, configs, gt_seg=None, gt_instance=None, gt_offset_y=None, gt_z=None,image_gt_segment=None,
                image_gt_instance=None, train=True):
        res = self.model(inputs, images_gt, configs)
        # images_gt: 用于 2d网络 监督信号
        image_gt_instance_h = res[0]
        image_gt_segment_h = res[1]
        homograph_matrix = res[2]
        pred, emb, offset_y, z = res[3]
        pred_2d, emb_2d = res[4]
        if train:
            ## 3d
            loss_seg = self.bce(pred, gt_seg) + self.iou_loss(torch.sigmoid(pred), gt_seg)
            loss_emb = self.poopoo(emb, gt_instance)
            loss_offset = self.bce_loss(gt_seg * torch.sigmoid(offset_y), gt_offset_y)
            loss_z = self.mse_loss(gt_seg * z, gt_z)
            loss_total = 3 * loss_seg + 0.5 * loss_emb
            loss_total = loss_total.unsqueeze(0)
            loss_offset = 60 * loss_offset.unsqueeze(0)
            loss_z = 30 * loss_z.unsqueeze(0)
            ## 2d
            loss_seg_2d = self.bce(pred_2d, image_gt_segment) + self.iou_loss(torch.sigmoid(pred_2d), image_gt_segment)
            loss_emb_2d = self.poopoo(emb_2d, image_gt_instance)
            loss_total_2d = 3 * loss_seg_2d + 0.5 * loss_emb_2d
            loss_total_2d = loss_total_2d.unsqueeze(0)

            homograph_matrix_inv = torch.inverse(homograph_matrix)
            # homograph_matrix_inv = homograph_matrix_inv.reshape(-1, 9)
            homograph_matrix_inv = F.normalize(homograph_matrix_inv, dim=(1, 2), p=2, eps=1e-6)  # H^-1
            # homograph_matrix_inv = homograph_matrix_inv.reshape()
            # homograph_matrix_inv = keh.normalize(homograph_matrix_inv.unsqueeze(0), mean=homograph_matrix_inv.mean(dim=(1, 2)), std=homograph_matrix_inv.var(dim=(1, 2))).squeeze(0)
            # homograph_matrix_inv = kgc.normalize_homography(homograph_matrix_inv,(pred_2d.shape[2], pred_2d.shape[3]), (pred_2d.shape[2], pred_2d.shape[3]))
            homograph_matrix_inv = kgc.denormalize_homography(homograph_matrix_inv,
                                                              (pred_2d.shape[2], pred_2d.shape[3]),
                                                              (pred_2d.shape[2], pred_2d.shape[3]))
            pred_2d_h_invs = kgt.warp_perspective(pred_2d, homograph_matrix_inv, configs.output_2d_shape)
            emb_2d_h_invs = kgt.warp_perspective(emb_2d, homograph_matrix_inv, configs.output_2d_shape)
            loss_seg_hg = self.bce(pred_2d_h_invs, image_gt_segment) + self.iou_loss(torch.sigmoid(pred_2d_h_invs),
                                                                                     image_gt_segment)
            loss_emb_hg = self.poopoo(emb_2d_h_invs, image_gt_instance)  # 计算2D嵌入向量损失
            loss_total_hg = 3 * loss_seg_hg + 0.5 * loss_emb_hg  # 计算hg总损失
            return pred, loss_total, loss_total_2d, loss_offset, loss_z, homograph_matrix, homograph_matrix_inv, loss_total_hg, loss_seg_hg, loss_emb_hg # 返回预测结果和损失
        else:
            return pred


def train_epoch(model, dataset, optimizer,scheduler, configs, epoch):
    # Last iter as mean loss of whole epoch
    model.train()
    losses_avg = {}
    '''image,image_gt_segment,image_gt_instance,ipm_gt_segment,ipm_gt_instance'''
    for idx, (  # image, ipm_gt_segment.float(), ipm_gt_instance.float(), ipm_gt_offset.float(), ipm_gt_z.float(), image_gt_segment.float(), image_gt_instance.float()
            input_data, image_gt, gt_seg_data, gt_emb_data, offset_y_data, z_data, image_gt_segment,
            image_gt_instance) in enumerate(
            dataset): #TODO: dataset __getitem__是否是在这里调用？
        # loss_back, loss_iter = forward_on_cuda(gpu, gt_data, input_data, loss, models)
        input_data = input_data.cuda()
        gt_seg_data = gt_seg_data.cuda()
        gt_emb_data = gt_emb_data.cuda()
        offset_y_data = offset_y_data.cuda()
        z_data = z_data.cuda()
        prediction, loss_total_bev, loss_total_2d, loss_offset, loss_z, hg_matrix, homograph_matrix_inv, loss_total_hg, loss_seg_hg, loss_emb_hg = model(input_data,
                                                                                                   image_gt,
                                                                                                   configs,
                                                                                                   gt_seg_data,
                                                                                                   gt_emb_data,
                                                                                                   offset_y_data,
                                                                                                   z_data,
                                                                                                   image_gt_segment,
                                                                                                   image_gt_instance)  # 正向传播
        loss_back_bev = loss_total_bev.mean()
        loss_back_2d = loss_total_2d.mean()
        loss_offset = loss_offset.mean()
        loss_z = loss_z.mean()
        loss_back_total = loss_back_bev + 0.5 * loss_back_2d + loss_offset + loss_z

        loss_total_hg = loss_total_hg.mean()
        loss_seg_hg = loss_seg_hg.mean()
        loss_emb_hg = loss_emb_hg.mean()
        ''' caclute loss '''

        optimizer.zero_grad()
        loss_back_total.backward()
        optimizer.step()
        if idx % 50 == 0:
            target = gt_seg_data.detach().cpu().numpy().ravel()
            pred = torch.sigmoid(prediction).detach().cpu().numpy().ravel()
            f1_bev_seg = f1_score((target > 0.5).astype(np.int64), (pred > 0.5).astype(np.int64), zero_division=1)
            print('| %3d | Hlr: %.10f | Blr: %.10f | 2d+3d: %f | F1: %f | Offset: %f | Z: %f | 3d: %f | 2d: %f | h: %f | hs: %f | he: %f |' % (
                    idx, scheduler.optimizer.param_groups[0]['lr'], scheduler.optimizer.param_groups[1]['lr'],
                    loss_back_total.item(),
                    f1_bev_seg, loss_offset.item(), loss_z.item(),
                    loss_back_bev.item(), loss_back_2d.item(), loss_total_hg.item(), loss_seg_hg.item(),
                    loss_emb_hg.item()))
        # if idx % 300 == 0:
        #     target = gt_seg_data.detach().cpu().numpy().ravel()
        #     pred = torch.sigmoid(prediction).detach().cpu().numpy().ravel()
        #     f1_bev_seg = f1_score((target > 0.5).astype(np.int64), (pred > 0.5).astype(np.int64), zero_division=1)
        #     loss_iter = {"BEV Loss": loss_back_bev.item(), 'offset loss': loss_offset.item(), 'z loss': loss_z.item(),
        #                     "F1_BEV_seg": f1_bev_seg}
        #     # losses_show = loss_iter
        #     print(idx, loss_iter)
        if idx != 0 and idx % 700 == 0:
            # print([i for i in hg_matrix[0].view(1, 9).squeeze(0).detach().cpu().numpy()])
            print('hm__: ', [i for i in hg_matrix[0].view(1, 9).squeeze(0).detach().cpu().numpy()])  # 原始matrix
            hg_mtxs_image = kgc.denormalize_homography(hg_matrix, configs.input_shape, configs.input_shape)
            hg_mtxs_image_gt = kgc.denormalize_homography(hg_matrix, configs.output_2d_shape, configs.output_2d_shape)
            print('hm_m: ',
                  [i for i in hg_mtxs_image[0].view(1, 9).squeeze(0).detach().cpu().numpy()])  # image_denormal
            print('hm_t: ',
                  [i for i in hg_mtxs_image_gt[0].view(1, 9).squeeze(0).detach().cpu().numpy()])  # image_gt_denormal
            print('hm_v: ', [i for i in
                             homograph_matrix_inv[0].view(1, 9).squeeze(0).detach().cpu().numpy()])  # image_gt_denormal

def worker_function(config_file, gpu_id, checkpoint_path=None):
    print('use gpu ids is'+','.join([str(i) for i in gpu_id]))
    configs = load_config_module(config_file)

    ''' models and optimizer '''
    model = configs.model()
    model = Combine_Model_and_Loss(model)
    if torch.cuda.is_available():
        model = model.cuda()
    params_hg_ids = list(map(id, model.model.hg.parameters()))
    params_hg = filter(lambda m: (id(m) in params_hg_ids) and m.requires_grad, model.parameters())
    params_not_hg = filter(lambda m: (id(m) not in params_hg_ids) and m.requires_grad, model.parameters())

    optimizer = configs.optimizer(params=[
        {'params': params_hg, **configs.optimizer_params_hg},
        {'params': params_not_hg, **configs.optimizer_params}
    ])  # 定义优化器
    model = torch.nn.DataParallel(model)
    # optimizer = configs.optimizer(filter(lambda p: p.requires_grad, model.parameters()), **configs.optimizer_params)
    scheduler = getattr(configs, "scheduler", CosineAnnealingLR)(optimizer, configs.epochs)
    if checkpoint_path:
        if getattr(configs, "load_optimizer", True):
            resume_training(checkpoint_path, model.module, optimizer, scheduler, configs.resume_scheduler)
        else:
            load_checkpoint(checkpoint_path, model.module, None)

    ''' dataset '''
    Dataset = getattr(configs, "train_dataset", None)
    if Dataset is None:
        Dataset = configs.training_dataset
    train_loader = DataLoader(Dataset(), **configs.loader_args, pin_memory=True)

    ''' get validation '''
    # if configs.with_validation:
    #     val_dataset = Dataset(**configs.val_dataset_args)
    #     val_loader = DataLoader(val_dataset, **configs.val_loader_args, pin_memory=True)
    #     val_loss = getattr(configs, "val_loss", loss)
    #     if eval_only:
    #         loss_mean = val_dp(model, val_loader, val_loss)
    #         print(loss_mean)
    #         return

    for epoch in range(configs.epochs):
        print('*' * 100, epoch)
        train_epoch(model, train_loader, optimizer, scheduler, configs, epoch)
        scheduler.step()
        save_model_dp(model, optimizer, scheduler, configs.model_save_path, 'ep%03d.pth' % epoch)
        save_model_dp(model, None, None, configs.model_save_path, 'latest.pth')


# TODO template config file.
if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    worker_function('/home/houzm/houzm/02_code/bev_lane_det-cnn/tools/openlane_config.py', 
                    gpu_id=[4, 5, 6, 7],
                    checkpoint_path='/home/houzm/houzm/03_model/bev_lane_det-cnn/openlane/train/0716_02/ep049.pth'
                    )
