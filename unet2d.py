from datetime import datetime
from functools import partial
import glob
from multiprocessing import Pool

# Disable multiprocesing for numpy/opencv. We already multiprocess ourselves, this would mean every subprocess produces
# even more threads which would lead to a lot of context switching, slowing things down a lot.
import os
os.environ["OMP_NUM_THREADS"] = "1"

import matplotlib.pyplot as plt

import pandas as pd
import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm
import scipy
import scipy.ndimage
import scipy.special
from scipy.spatial.transform import Rotation as R

from lyft_dataset_sdk.lyftdataset import LyftDataset

# pzr
from lyft_dataset_sdk.lyftdataset import LyftDatasetExplorer

from lyft_dataset_sdk.utils.data_classes import LidarPointCloud, Box, Quaternion
from lyft_dataset_sdk.utils.geometry_utils import view_points, transform_matrix

import os

# os.symlink("/media/pzr/软件/kaggle/3D/train_images", 'images')
# os.symlink("/media/pzr/软件/kaggle/3D/train_maps", 'maps')
# os.symlink("/media/pzr/软件/kaggle/3D/train_lidar", 'lidar')

# Our code will generate data, visualization and model checkpoints, they will be persisted to disk in this folder
DATA_PATH = '/media/pzr/软件/kaggle/3D/train_data'
ARTIFACTS_FOLDER = "/media/pzr/软件/kaggle/3D/artifacts/"

# 这里我软链接建立了但是还是链接不上map_raster_palo_alto.png文件，所以改了一下map.json 的最后一行为"filename": "test_maps/map_raster_palo_alto.png", "category": "semantic_prior"}]
# 在LyftDataset源码可以看到里面会读train_data里面的所有json 文件
# 所有修改过的 或者 我添加的代码在前面用pzr表示
level5data = LyftDataset(data_path='.', json_path=DATA_PATH, verbose=True)
os.makedirs(ARTIFACTS_FOLDER, exist_ok=True)

# pzr 这个Explorer 可以输出一些数据信息
dataHelper = LyftDatasetExplorer(level5data)
dataHelper.list_categories()

# Category stats
# animal                      n=  186, width= 0.36±0.12, len= 0.73±0.19, height= 0.51±0.16, lw_aspect= 2.16±0.56
# bicycle                     n=20928, width= 0.63±0.24, len= 1.76±0.29, height= 1.44±0.37, lw_aspect= 3.20±1.17
# bus                         n= 8729, width= 2.96±0.24, len=12.34±3.41, height= 3.44±0.31, lw_aspect= 4.17±1.10
# car                         n=534911, width= 1.93±0.16, len= 4.76±0.53, height= 1.72±0.24, lw_aspect= 2.47±0.22
# emergency_vehicle           n=  132, width= 2.45±0.43, len= 6.52±1.44, height= 2.39±0.59, lw_aspect= 2.66±0.28
# motorcycle                  n=  818, width= 0.96±0.20, len= 2.35±0.22, height= 1.59±0.16, lw_aspect= 2.53±0.50
# other_vehicle               n=33376, width= 2.79±0.30, len= 8.20±1.71, height= 3.23±0.50, lw_aspect= 2.93±0.53
# pedestrian                  n=24935, width= 0.77±0.14, len= 0.81±0.17, height= 1.78±0.16, lw_aspect= 1.06±0.20
# truck                       n=14164, width= 2.84±0.32, len=10.24±4.09, height= 3.44±0.62, lw_aspect= 3.56±1.25

dataHelper.list_attributes()

# object_action_abnormal_or_traffic_violation: 2
# object_action_driving_straight_forward: 244805
# object_action_gliding_on_wheels: 165
# object_action_lane_change_left: 1463
# object_action_lane_change_right: 1370
# object_action_left_turn: 5074
# object_action_loss_of_control: 1
# object_action_other_motion: 582
# object_action_parked: 257939
# object_action_reversing: 278
# object_action_right_turn: 6694
# object_action_running: 621
# object_action_sitting: 586
# object_action_standing: 5332
# object_action_stopped: 94970
# object_action_u_turn: 407
# object_action_walking: 17890

dataHelper.list_scenes()

# 180 个 scene , 之后根据不同的scene split train and validation


###



print (level5data)

# 9 category,
# 18 attribute,  object_action_abnormal_or_traffic_violation: 2 显示一些车辆、道路的信息
# 4 visibility,
# 18421 instance,
# 10 sensor,
# 148 calibrated_sensor,
# 177789 ego_pose,
# 180 log,
# 180 scene,   180个场景 （不同地区的信息 雷达、 图像）
# 22680 sample,
# 189504 sample_data,
# 638179 sample_annotation,
# 1 map,
# Done loading in 33.3 seconds.

# https://www.kaggle.com/tarunpaparaju/lyft-competition-understanding-the-data 里有对这些名词的解释
# https://www.kaggle.com/c/3d-object-detection-for-autonomous-vehicles/discussion/110257#latest-636505
# 里面可视化了所有的文件里面包含的信息 各个json可以看作数据库中的表，通过主键 外键 进行关联

classes = ["car", "motorcycle", "bus", "bicycle", "truck", "pedestrian", "other_vehicle", "animal", "emergency_vehicle"]

print (level5data.scene[1])

# https://www.kaggle.com/c/3d-object-detection-for-autonomous-vehicles/discussion/110257#latest-636505 可以看到scene的结构 以及包含的键 与变量 与输出吻合
# {'description': '', 'first_sample_token': '3b30673b9d944ec6058ef5a8debb4c0a6fe075bca7076e06cf42a2bc10dc446e', 'log_token': '0a6839d6ee6804113bb5591ed99cc70ad883d0cff396e3aec5e76e718771b30e', 'name': 'host-a006-lidar0-1236037883198113706-1236037908098879296', 'token': '0a6839d6ee6804113bb5591ed99cc70ad883d0cff396e3aec5e76e718771b30e', 'last_sample_token': '5ec01e4634ca91751311eaafb45a9196ba8616bf05edc85593aff158db653a34', 'nbr_samples': 126}

# get Returns a record from table   ------- sample 是 表名，对应sample.json
# 通过scene 当中的 first_sample_token 得到 timestamp 与 record (scene) 一起放到一个列表里

records = [(level5data.get('sample', record['first_sample_token'])['timestamp'],
            record) for record in level5data.scene]

print (len(records)) # 180   180 个scene
print (records[:2])

# [(1557858039302414.8, {'log_token': 'da4ed9e02f64c544f4f1f10c6738216dcb0e6b0d50952e158e5589854af9f100', 'first_sample_token': '24b0962e44420e6322de3f25d9e4e5cc3c7a348ec00bfa69db21517e4ca92cc8', 'name': 'host-a101-lidar0-1241893239199111666-1241893264098084346', 'description': '', 'last_sample_token': '2346756c83f6ae8c4d1adec62b4d0d31b62116d2e1819e96e9512667d15e7cec', 'nbr_samples': 126, 'token': 'da4ed9e02f64c544f4f1f10c6738216dcb0e6b0d50952e158e5589854af9f100'}),
# (1552002683300887.5, {'description': '', 'first_sample_token': '3b30673b9d944ec6058ef5a8debb4c0a6fe075bca7076e06cf42a2bc10dc446e', 'log_token': '0a6839d6ee6804113bb5591ed99cc70ad883d0cff396e3aec5e76e718771b30e', 'name': 'host-a006-lidar0-1236037883198113706-1236037908098879296', 'token': '0a6839d6ee6804113bb5591ed99cc70ad883d0cff396e3aec5e76e718771b30e', 'last_sample_token': '5ec01e4634ca91751311eaafb45a9196ba8616bf05edc85593aff158db653a34', 'nbr_samples': 126})]


# 下面是看一下train.csv 中间包含638179个已经标定好的样本 即一张图片中可能有多个object 每个object对应一行



# train = pd.read_csv(DATA_PATH + 'train.csv')
# sample_submission = pd.read_csv(DATA_PATH + 'sample_submission.csv')
#
# # Group data by object category
#
# object_columns = ['sample_id', 'object_id', 'center_x', 'center_y', 'center_z',
#                   'width', 'length', 'height', 'yaw', 'class_name']
# objects = []
# for sample_id, ps in tqdm(train.values[:]):
#     object_params = ps.split()
#     n_objects = len(object_params)
#     for i in range(n_objects // 8):
#         x, y, z, w, l, h, yaw, c = tuple(object_params[i * 8: (i + 1) * 8])
#         objects.append([sample_id, i, x, y, z, w, l, h, yaw, c])
#
# train_objects = pd.DataFrame(
#     objects,
#     columns = object_columns
# )
#
# numerical_cols = ['object_id', 'center_x', 'center_y', 'center_z', 'width', 'length', 'height', 'yaw']
# train_objects[numerical_cols] = np.float32(train_objects[numerical_cols].values)
#
# print (train_objects.shape)  # (638179, 10) 638179 个annotation 包含 sample_id, i, x, y, z, w, l, h, yaw, c
#
# print (train_objects.head())

# 这里处理json（表） 的方式跟上面一样

entries = []

for start_time, record in sorted(records):
    # 通过 first_sample_token 取得sample表
    # sample表示一个Scene中的一个snapshot(相当于一个瞬间的记录，每个scene有126个sample，所以一共有126 * 180 = 22680 个sample
    start_time = level5data.get('sample', record['first_sample_token'])['timestamp'] / 1000000 # start_time: float
    token = record['token']
    name = record['name']
    date = datetime.utcfromtimestamp(start_time)
    host = "-".join(record['name'].split("-")[:2])
    first_sample_token = record["first_sample_token"]

    entries.append((host, name, date, token, first_sample_token))

df = pd.DataFrame(entries, columns=["host", "scene_name", "date",
                                    "scene_token", "first_sample_token"])

# host_count_df = df.groupby("host")['scene_token'].count()
host_count_df = df.groupby("host")['scene_token'].count()
print(df.groupby("host")['scene_token'])
print(host_count_df)

# 根据host（应该就是车的代号）group , scene_token的数量代表了每辆车的scene数量，即每一辆车开了多少scenes
# scene - Consists of 25-45 seconds of a car's journey in a given environment.
# Each scence is composed of many samples.
# host
# host-a004    42
# host-a005     1
# host-a006     3
# host-a007    26
# host-a008     5
# host-a009     9
# host-a011    51
# host-a012     2
# host-a015     6
# host-a017     3
# host-a101    20
# host-a102    12

# 分配数据集
validation_hosts = ["host-a007", "host-a008", "host-a009"]

validation_df = df[df["host"].isin(validation_hosts)]
vi = validation_df.index
train_df = df[~df.index.isin(vi)]

print(len(train_df), len(validation_df), "train/validation split scene counts")

print(train_df.first_sample_token.values[0])
# Creating input and targets
# Let's load the first sample in the train set. We can use that to test the functions
# we'll define next that transform the data to the format we want to input into the model we are training.

sample_token = train_df.first_sample_token.values[56]

# sample_token : cea0bba4b425537cca52b17bf81569a20da1ca6d359f33227f0230d59d9d2881
# 在sample 表里面根据sample_stoken拿到sample
sample = level5data.get("sample", sample_token)

# 拿到上方雷达token 上方雷达数据可视化在 https://www.kaggle.com/tarunpaparaju/lyft-competition-understanding-the-data
sample_lidar_token = sample["data"]["LIDAR_TOP"]

# 在sample_data 表中拿到lidar data , sample_data中也有 picture data
lidar_data = level5data.get("sample_data", sample_lidar_token)
lidar_filepath = level5data.get_sample_data_path(sample_lidar_token)

print (lidar_filepath)
print (os.path.exists(lidar_filepath))

# 在ego_pose 表中拿到 lidar_data 对应的ego_pose
ego_pose = level5data.get("ego_pose", lidar_data["ego_pose_token"])

# 拿到calibrated_sensor 这些token 都可以在https://www.kaggle.com/c/3d-object-detection-for-autonomous-vehicles/discussion/110257#latest-636505的sample_data.json里面找到
calibrated_sensor = level5data.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])

# 坐标系的转换 从车坐标系转到世界坐标系
# Homogeneous transformation matrix from car frame to world frame.
global_from_car = transform_matrix(ego_pose['translation'],
                                   Quaternion(ego_pose['rotation']), inverse=False)
# 从雷达转到车坐标系
# Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
car_from_sensor = transform_matrix(calibrated_sensor['translation'], Quaternion(calibrated_sensor['rotation']),
                                    inverse=False)

# 拿到的坐标是相机坐标系的坐标  所以不是以车为中心
lidar_pointcloud = LidarPointCloud.from_file(lidar_filepath)

# 将相机坐标系 转到 车的坐标系，  点就以原点为中心  可以通过直方图看出来，  X代表在车的X轴上的坐标 Y代表Y轴的坐标， 可以看到XY都以0为中心
# The lidar pointcloud is defined in the sensor's reference frame.
# We want it in the car's reference frame, so we transform each point
lidar_pointcloud.transform(car_from_sensor)
print (lidar_pointcloud.nbr_points())

print (lidar_pointcloud.points[0])
# [-97.33626   -95.891556  -95.581436  ...  -4.3027453  -4.3097425
#   -4.308989 ]

print (lidar_pointcloud.points[1])
# [ 8.948838    9.516727   10.167154   ...  0.17457172  0.19416472
#   0.21448715]

print (lidar_pointcloud.points[2])
# [11.900117   11.759933   11.736123   ...  0.15442245  0.15207818
#   0.15207745]
# lidar_pointcloud shape (4, 61908)

print (lidar_pointcloud.points[3])
# [100. 100. 100. ... 100. 100. 100.]
# lidar_pointcloud shape (4, 61908)


# A sanity check, the points should be centered around 0 in car space.
plt.hist(lidar_pointcloud.points[0], alpha=0.5, bins=30, label="X")
plt.hist(lidar_pointcloud.points[1], alpha=0.5, bins=30, label="Y")
plt.legend()
plt.xlabel("Distance from car along axis")
plt.ylabel("Amount of points")
plt.show()


def create_transformation_matrix_to_voxel_space(shape, voxel_size, offset):
    """
    Constructs a transformation matrix given an output voxel shape such that (0,0,0) ends up in the center.
    Voxel_size defines how large every voxel is in world coordinate, (1,1,1) would be the same as Minecraft voxels.

    An offset per axis in world coordinates (metric) can be provided, this is useful for Z (up-down) in lidar points.
    """

    shape, voxel_size, offset = np.array(shape), np.array(voxel_size), np.array(offset)

    tm = np.eye(4, dtype=np.float32)
    # print (tm)
    translation = shape / 2 + offset / voxel_size

    tm = tm * np.array(np.hstack((1 / voxel_size, [1])))
    tm[:3, 3] = np.transpose(translation)
    return tm

def transform_points(points, transf_matrix):
    """
    Transform (3,N) or (4,N) points using transformation matrix.
    """
    if points.shape[0] not in [3, 4]:
        raise Exception("Points input should be (3,N) or (4,N) shape, received {}".format(points.shape))
    return transf_matrix.dot(np.vstack((points[:3, :], np.ones(points.shape[1]))))[:3, :]


# Let's try it with some example values
tm = create_transformation_matrix_to_voxel_space(shape=(100, 100, 4), voxel_size=(0.5, 0.5, 0.5), offset=(0, 0, 0.5))
p = transform_points(np.array([[10, 10, 0, 0, 0], [10, 5, 0, 0, 0], [0, 0, 0, 2, 0]], dtype=np.float32), tm)
print(p)


def car_to_voxel_coords(points, shape, voxel_size, z_offset=0):
    if len(shape) != 3:
        raise Exception("Voxel volume shape should be 3 dimensions (x,y,z)")

    if len(points.shape) != 2 or points.shape[0] not in [3, 4]:
        raise Exception("Input points should be (3,N) or (4,N) in shape, found {}".format(points.shape))

    tm = create_transformation_matrix_to_voxel_space(shape, voxel_size, (0, 0, z_offset))
    p = transform_points(points, tm)
    return p


def create_voxel_pointcloud(points, shape, voxel_size=(0.5, 0.5, 1), z_offset=0):
    points_voxel_coords = car_to_voxel_coords(points.copy(), shape, voxel_size, z_offset)
    points_voxel_coords = points_voxel_coords[:3].transpose(1, 0)
    points_voxel_coords = np.int0(points_voxel_coords)

    bev = np.zeros(shape, dtype=np.float32)
    bev_shape = np.array(shape)

    within_bounds = (np.all(points_voxel_coords >= 0, axis=1) * np.all(points_voxel_coords < bev_shape, axis=1))

    points_voxel_coords = points_voxel_coords[within_bounds]
    coord, count = np.unique(points_voxel_coords, axis=0, return_counts=True)

    # Note X and Y are flipped:
    bev[coord[:, 1], coord[:, 0], coord[:, 2]] = count

    return bev


def normalize_voxel_intensities(bev, max_intensity=16):
    return (bev / max_intensity).clip(0, 1)


voxel_size = (0.4, 0.4, 1.5)
z_offset = -2.0
bev_shape = (336, 336, 3)

bev = create_voxel_pointcloud(lidar_pointcloud.points, bev_shape, voxel_size=voxel_size, z_offset=z_offset)

# So that the values in the voxels range from 0,1 we set a maximum intensity.
bev = normalize_voxel_intensities(bev)


plt.figure(figsize=(16,8))
plt.imshow(bev)
plt.show()

boxes = level5data.get_boxes(sample_lidar_token)

print(boxes[1].bottom_corners())
print(bev.shape[:3])
target_im = np.zeros(bev.shape[:3], dtype=np.uint8)


def move_boxes_to_car_space(boxes, ego_pose):
    """
    Move boxes from world space to car space.
    Note: mutates input boxes.
    """
    translation = -np.array(ego_pose['translation'])
    rotation = Quaternion(ego_pose['rotation']).inverse

    for box in boxes:
        # Bring box to car space
        box.translate(translation)
        box.rotate(rotation)


def scale_boxes(boxes, factor):
    """
    Note: mutates input boxes
    """
    for box in boxes:
        box.wlh = box.wlh * factor


def draw_boxes(im, voxel_size, boxes, classes, z_offset=0.0):
    for box in boxes:
        # We only care about the bottom corners
        corners = box.bottom_corners()
        corners_voxel = car_to_voxel_coords(corners, im.shape, voxel_size, z_offset).transpose(1, 0)
        corners_voxel = corners_voxel[:, :2]  # Drop z coord

        class_color = classes.index(box.name) + 1

        if class_color == 0:
            raise Exception("Unknown class: {}".format(box.name))

        cv2.drawContours(im, np.int0([corners_voxel]), 0, (class_color, class_color, class_color), -1)


move_boxes_to_car_space(boxes, ego_pose)
scale_boxes(boxes, 0.8)
draw_boxes(target_im, voxel_size, boxes, classes, z_offset=z_offset)

plt.figure(figsize=(8, 8))
plt.imshow((target_im > 0).astype(np.float32), cmap='Set2')
plt.show()


def visualize_lidar_of_sample(sample_token, axes_limit=80):
    sample = level5data.get("sample", sample_token)
    sample_lidar_token = sample["data"]["LIDAR_TOP"]
    level5data.render_sample_data(sample_lidar_token, axes_limit=axes_limit)


# Don't worry about it being mirrored.
visualize_lidar_of_sample(sample_token)

del bev, lidar_pointcloud, boxes

# Some hyperparameters we'll need to define for the system
voxel_size = (0.4, 0.4, 1.5)
z_offset = -2.0
bev_shape = (336, 336, 3)

# We scale down each box so they are more separated when projected into our coarse voxel space.
box_scale = 0.8


# "bev" stands for birds eye view
train_data_folder = os.path.join(ARTIFACTS_FOLDER, "bev_train_data")
validation_data_folder = os.path.join(ARTIFACTS_FOLDER, "./bev_validation_data")

NUM_WORKERS = os.cpu_count() * 3


def prepare_training_data_for_scene(first_sample_token, output_folder, bev_shape, voxel_size, z_offset, box_scale):
    """
    Given a first sample token (in a scene), output rasterized input volumes and targets in birds-eye-view perspective.

    """
    sample_token = first_sample_token

    while sample_token:

        sample = level5data.get("sample", sample_token)

        sample_lidar_token = sample["data"]["LIDAR_TOP"]
        lidar_data = level5data.get("sample_data", sample_lidar_token)
        lidar_filepath = level5data.get_sample_data_path(sample_lidar_token)

        ego_pose = level5data.get("ego_pose", lidar_data["ego_pose_token"])
        calibrated_sensor = level5data.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])

        global_from_car = transform_matrix(ego_pose['translation'],
                                           Quaternion(ego_pose['rotation']), inverse=False)

        car_from_sensor = transform_matrix(calibrated_sensor['translation'], Quaternion(calibrated_sensor['rotation']),
                                           inverse=False)

        try:
            lidar_pointcloud = LidarPointCloud.from_file(lidar_filepath)
            lidar_pointcloud.transform(car_from_sensor)
        except Exception as e:
            print("Failed to load Lidar Pointcloud for {}: {}:".format(sample_token, e))
            sample_token = sample["next"]
            continue

        bev = create_voxel_pointcloud(lidar_pointcloud.points, bev_shape, voxel_size=voxel_size, z_offset=z_offset)
        bev = normalize_voxel_intensities(bev)

        boxes = level5data.get_boxes(sample_lidar_token)

        target = np.zeros_like(bev)

        move_boxes_to_car_space(boxes, ego_pose)
        scale_boxes(boxes, box_scale)
        draw_boxes(target, voxel_size, boxes=boxes, classes=classes, z_offset=z_offset)

        bev_im = np.round(bev * 255).astype(np.uint8)
        target_im = target[:, :, 0]  # take one channel only

        cv2.imwrite(os.path.join(output_folder, "{}_input.png".format(sample_token)), bev_im)
        cv2.imwrite(os.path.join(output_folder, "{}_target.png".format(sample_token)), target_im)

        sample_token = sample["next"]

'''
for df, data_folder in [(train_df, train_data_folder), (validation_df, validation_data_folder)]:
    print("Preparing data into {} using {} workers".format(data_folder, NUM_WORKERS))
    first_samples = df.first_sample_token.values

    os.makedirs(data_folder, exist_ok=True)

    process_func = partial(prepare_training_data_for_scene,
                           output_folder=data_folder, bev_shape=bev_shape, voxel_size=voxel_size, z_offset=z_offset,
                           box_scale=box_scale)

    pool = Pool(NUM_WORKERS)
    for _ in tqdm(pool.imap_unordered(process_func, first_samples), total=len(first_samples)):
        pass
    pool.close()
    del pool
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


class BEVImageDataset(torch.utils.data.Dataset):
    def __init__(self, input_filepaths, target_filepaths, map_filepaths=None):
        self.input_filepaths = input_filepaths
        self.target_filepaths = target_filepaths
        self.map_filepaths = map_filepaths

        if map_filepaths is not None:
            assert len(input_filepaths) == len(map_filepaths)

        assert len(input_filepaths) == len(target_filepaths)

    def __len__(self):
        return len(self.input_filepaths)

    def __getitem__(self, idx):
        input_filepath = self.input_filepaths[idx]
        target_filepath = self.target_filepaths[idx]

        sample_token = input_filepath.split("/")[-1].replace("_input.png", "")

        im = cv2.imread(input_filepath, cv2.IMREAD_UNCHANGED)

        if self.map_filepaths:
            map_filepath = self.map_filepaths[idx]
            map_im = cv2.imread(map_filepath, cv2.IMREAD_UNCHANGED)
            im = np.concatenate((im, map_im), axis=2)

        target = cv2.imread(target_filepath, cv2.IMREAD_UNCHANGED)

        im = im.astype(np.float32) / 255
        target = target.astype(np.int64)

        im = torch.from_numpy(im.transpose(2, 0, 1))
        target = torch.from_numpy(target)

        return im, target, sample_token


input_filepaths = sorted(glob.glob(os.path.join(train_data_folder, "*_input.png")))
target_filepaths = sorted(glob.glob(os.path.join(train_data_folder, "*_target.png")))

train_dataset = BEVImageDataset(input_filepaths, target_filepaths)

im, target, sample_token = train_dataset[1]
im = im.numpy()
target = target.numpy()

plt.figure(figsize=(16, 8))

target_as_rgb = np.repeat(target[..., None], 3, 2)
# Transpose the input volume CXY to XYC order, which is what matplotlib requires.
plt.imshow(np.hstack((im.transpose(1, 2, 0)[..., :3], target_as_rgb)))
plt.title(sample_token)
plt.show()

visualize_lidar_of_sample(sample_token)

# This implementation was copied from https://github.com/jvanvugt/pytorch-unet, it is MIT licensed.

class UNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        n_classes=2,
        depth=5,
        wf=6,
        padding=False,
        batch_norm=False,
        up_mode='upconv',
    ):
        """
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597
        Using the default arguments will yield the exact version used
        in the original paper
        Args:
            in_channels (int): number of input channels
            n_classes (int): number of output channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
                            This may introduce artifacts
            batch_norm (bool): Use BatchNorm after layers with an
                               activation function
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for
                           learned upsampling.
                           'upsample' will use bilinear upsampling.
        """
        super(UNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(
                UNetConvBlock(prev_channels, 2 ** (wf + i), padding, batch_norm)
            )
            prev_channels = 2 ** (wf + i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(
                UNetUpBlock(prev_channels, 2 ** (wf + i), up_mode, padding, batch_norm)
            )
            prev_channels = 2 ** (wf + i)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = F.max_pool2d(x, 2)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        return self.last(x)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        block.append(nn.Conv2d(out_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_size, out_size, kernel_size=1),
            )

        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[
            :, :, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1])
        ]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out


def get_unet_model(in_channels=3, num_output_classes=2):
    model = UNet(in_channels=in_channels, n_classes=num_output_classes, wf=5, depth=4, padding=True, up_mode='upsample')

    # Optional, for multi GPU training and inference
    model = nn.DataParallel(model)
    return model


def visualize_predictions(input_image, prediction, target, n_images=2, apply_softmax=True):
    """
    Takes as input 3 PyTorch tensors, plots the input image, predictions and targets.
    """
    # Only select the first n images
    prediction = prediction[:n_images]
    target = target[:n_images]
    input_image = input_image[:n_images]

    prediction = prediction.detach().cpu().numpy()
    if apply_softmax:
        prediction = scipy.special.softmax(prediction, axis=1)
    class_one_preds = np.hstack(1 - prediction[:, 0])

    target = np.hstack(target.detach().cpu().numpy())

    class_rgb = np.repeat(class_one_preds[..., None], 3, axis=2)
    class_rgb[..., 2] = 0
    class_rgb[..., 1] = target

    input_im = np.hstack(input_image.cpu().numpy().transpose(0, 2, 3, 1))

    if input_im.shape[2] == 3:
        input_im_grayscale = np.repeat(input_im.mean(axis=2)[..., None], 3, axis=2)
        overlayed_im = (input_im_grayscale * 0.6 + class_rgb * 0.7).clip(0, 1)
    else:
        input_map = input_im[..., 3:]
        overlayed_im = (input_map * 0.6 + class_rgb * 0.7).clip(0, 1)

    thresholded_pred = np.repeat(class_one_preds[..., None] > 0.5, 3, axis=2)

    fig = plt.figure(figsize=(12, 26))
    plot_im = np.vstack([class_rgb, input_im[..., :3], overlayed_im, thresholded_pred]).clip(0, 1).astype(np.float32)
    plt.imshow(plot_im)
    plt.axis("off")
    plt.show()


# We weigh the loss for the 0 class lower to account for (some of) the big class imbalance.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class_weights = torch.from_numpy(np.array([0.2] + [1.0]*len(classes), dtype=np.float32))
class_weights = class_weights.to(device)

batch_size = 3
epochs = 30  # Note: We may be able to train for longer and expect better results, the reason this number is low is to keep the runtime short.

model = get_unet_model(num_output_classes=len(classes) + 1)
model = model.to(device)

optim = torch.optim.Adam(model.parameters(), lr=1e-3)
dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True, num_workers=os.cpu_count() * 2)

all_losses = []

for epoch in range(1, epochs + 1):
    print("Epoch", epoch)

    epoch_losses = []
    progress_bar = tqdm(dataloader)

    for ii, (X, target, sample_ids) in enumerate(progress_bar):
        X = X.to(device)  # [N, 3, H, W]
        target = target.to(device)  # [N, H, W] with class indices (0, 1)
        prediction = model(X)  # [N, 2, H, W]
        loss = F.cross_entropy(prediction, target, weight=class_weights)

        optim.zero_grad()
        loss.backward()
        optim.step()

        epoch_losses.append(loss.detach().cpu().numpy())

        #if ii == 0:
        #    visualize_predictions(X, prediction, target)

    print("Loss:", np.mean(epoch_losses))
    all_losses.extend(epoch_losses)

    checkpoint_filename = "unet_checkpoint_epoch_{}.pth".format(epoch)
    checkpoint_filepath = os.path.join(ARTIFACTS_FOLDER, checkpoint_filename)
    torch.save(model.state_dict(), checkpoint_filepath)

plt.figure(figsize=(12, 12))
plt.plot(all_losses, alpha=0.75)
plt.show()


input_filepaths = sorted(glob.glob(os.path.join(validation_data_folder, "*_input.png")))
target_filepaths = sorted(glob.glob(os.path.join(validation_data_folder, "*_target.png")))

batch_size=3
validation_dataset = BEVImageDataset(input_filepaths, target_filepaths)
validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size, shuffle=False, num_workers=os.cpu_count())


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_unet_model(num_output_classes=1+len(classes))
model = model.to(device)

epoch_to_load=30
checkpoint_filename = "unet_checkpoint_epoch_{}.pth".format(epoch_to_load)
checkpoint_filepath = os.path.join(ARTIFACTS_FOLDER, checkpoint_filename)
model.load_state_dict(torch.load(checkpoint_filepath))

progress_bar = tqdm(validation_dataloader)

targets = np.zeros((len(target_filepaths), 336, 336), dtype=np.uint8)

# We quantize to uint8 here to conserve memory. We're allocating >20GB of memory otherwise.
predictions = np.zeros((len(target_filepaths), 1 + len(classes), 336, 336), dtype=np.uint8)

sample_tokens = []
all_losses = []

with torch.no_grad():
    model.eval()
    for ii, (X, target, batch_sample_tokens) in enumerate(progress_bar):

        offset = ii * batch_size
        targets[offset:offset + batch_size] = target.numpy()
        sample_tokens.extend(batch_sample_tokens)

        X = X.to(device)  # [N, 1, H, W]
        target = target.to(device)  # [N, H, W] with class indices (0, 1)
        prediction = model(X)  # [N, 2, H, W]
        loss = F.cross_entropy(prediction, target, weight=class_weights)
        all_losses.append(loss.detach().cpu().numpy())

        prediction = F.softmax(prediction, dim=1)

        prediction_cpu = prediction.cpu().numpy()
        predictions[offset:offset + batch_size] = np.round(prediction_cpu * 255).astype(np.uint8)

        # Visualize the first prediction
        if ii == 0:
            visualize_predictions(X, prediction, target, apply_softmax=False)

print("Mean loss:", np.mean(all_losses))





















