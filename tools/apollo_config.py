import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from loader.bev_road.apollo_data import Apollo_dataset_with_offset, Apollo_dataset_with_offset_val
from models.model.single_camera_bev import BEV_LaneDet

def get_camera_matrix(cam_pitch,cam_height):
    proj_g2c = np.array([[1,                             0,                              0,          0],
                        [0, np.cos(np.pi / 2 + cam_pitch), -np.sin(np.pi / 2 + cam_pitch), cam_height],
                        [0, np.sin(np.pi / 2 + cam_pitch),  np.cos(np.pi / 2 + cam_pitch),          0],
                        [0,                             0,                              0,          1]])

    camera_K = np.array([[2015., 0., 960.],
                         [0., 2015., 540.],
                         [0., 0., 1.]])

    return proj_g2c, camera_K


''' data split '''
# illus_chg rare_subset
train_json_paths = '/home/houzm/datasets/apollo-3d-lane-synthetic/3D_Lane_Synthetic_Dataset-master/data_splits/standard/train.json'
test_json_paths = '/home/houzm/datasets/apollo-3d-lane-synthetic/3D_Lane_Synthetic_Dataset-master/data_splits/standard/test.json'
data_base_path = '/home/houzm/datasets/apollo-3d-lane-synthetic/Apollo_Sim_3D_Lane_Release'

model_save_path = "/home/houzm/houzm/03_model/bev_lane_det-cnn/apollo/train/0810_standard/"

input_shape = (576, 1024) # height width
output_2d_shape = (144, 256)

''' BEV range '''
x_range = (3, 103)
y_range = (-12, 12)
meter_per_pixel = 0.5  # grid size
bev_shape = (int((x_range[1] - x_range[0]) / meter_per_pixel), int((y_range[1] - y_range[0]) / meter_per_pixel))

loader_args = dict(
    batch_size=32,
    num_workers=12,
    shuffle=True
)

''' virtual camera config '''
camera_ext_virtual, camera_K_virtual = get_camera_matrix(0.04325083977888603, 1.7860000133514404)  # a random parameter
vc_config = {}
vc_config['use_virtual_camera'] = True
vc_config['vc_intrinsic'] = camera_K_virtual
vc_config['vc_extrinsics'] = np.linalg.inv(camera_ext_virtual)
vc_config['vc_image_shape'] = (1920, 1080)

''' model '''

# load_optimizer = False
load_optimizer = True

resume_scheduler = True
# resume_scheduler = False
def model():
    return BEV_LaneDet(bev_shape=bev_shape, output_2d_shape=output_2d_shape, train=True)


''' optimizer '''
epochs = 200
optimizer = AdamW
optimizer_params_hg = dict(
    lr=1e-1, betas=(0.9, 0.999), eps=1e-8,
    weight_decay=1e-2, amsgrad=False
)
optimizer_params = dict(
    lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
    weight_decay=1e-2, amsgrad=False
)
scheduler = CosineAnnealingLR

train_trans = A.Compose([
    A.Resize(height=input_shape[0], width=input_shape[1]),  # 调整图像大小为指定的输入形状
    A.MotionBlur(p=0.2),  # 随机应用运动模糊效果，概率为 0.2
    A.RandomBrightnessContrast(),  # 随机调整图像的亮度和对比度
    A.ColorJitter(p=0.1),  # 随机应用颜色增强，概率为 0.1
    A.Normalize(),  # 标准化图像
    ToTensorV2()  # 将图像转换为张量
])


def train_dataset():
    train_data = Apollo_dataset_with_offset(train_json_paths, data_base_path,
                                            x_range, y_range, meter_per_pixel,
                                            train_trans, output_2d_shape, vc_config)

    return train_data


def val_dataset():
    trans_image = A.Compose([
        A.Resize(height=input_shape[0], width=input_shape[1]),
        A.Normalize(),
        ToTensorV2()])
    val_data = Apollo_dataset_with_offset_val(test_json_paths, data_base_path,
                                              trans_image, vc_config)
    return val_data
