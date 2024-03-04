import os
import random
import numpy as np
import torch
# from plyfile import PlyData
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from time import time
import tqdm

from sklearn.neighbors import KDTree

import numpy as np

def farthest_point_sampling(point_cloud, num_points=10000):
    """
    使用FPS算法从点云中采样指定数量的点

    参数：
    point_cloud: numpy数组，形状为(N, 6)，其中N为点云中点的数量，最后一维包含了点的xyz坐标和其他信息
    num_points: 要采样的点的数量

    返回值：
    numpy数组，形状为(num_points, 6)，包含了采样的点的信息
    """

    # 提取点的xyz坐标
    xyz = point_cloud[:, :3]

    # 随机选择一个点作为第一个采样点
    sampled_indices = [np.random.randint(len(point_cloud))]
    distances = np.linalg.norm(xyz - xyz[sampled_indices[0]], axis=1)

    # 使用FPS算法逐步选择剩余的采样点
    for _ in range(1, num_points):
        farthest_index = np.argmax(distances)
        sampled_indices.append(farthest_index)
        distances = np.minimum(distances, np.linalg.norm(xyz - xyz[farthest_index], axis=1))

    # 根据采样的索引提取采样点
    sampled_points = point_cloud[sampled_indices]

    return sampled_points



class CarlaDataset(Dataset):
    # label_weights = np.random.normal(size=5)

    def __init__(self, carla_dir, transform=None, split='train', proportion=[0.7, 0.2, 0.1],
                 num_classes=5, sample_rate=0.1, numpoints=1024 * 10, need_speed=True, chanel_num = 3,
                 block_size=1.0, resample=False,random_sample=True):
        
        self.split = split  # 区分训练集或者测试集（当数据按文件划分后，可以用whole读取所有数据
        self.proportion = proportion  # 数据划分比例
        self.random_sample = random_sample
        # rootpath = os.path.abspath('..')
        self.num_classes = num_classes  # 语义类别数
        self.block_size = block_size  # 用于重采样的重采样块大小
        # self.carla_dir = os.path.join(carla_dir)  # 数据路径
        self.carla_dir = carla_dir
        self.transform = transform # 用于数据强化，主要是旋转、裁剪数据（目前尚未使用
        self.label_weights = np.random.normal(size=num_classes)  # 用于记录数据分布（指各个语义标签的数据占总体数据的比例）
        
        self.numpoints = numpoints  # 单帧中采样的点数
        all_file = os.listdir(self.carla_dir)  # 用于记录数据量
        self.need_speed = need_speed  # 用于区分是否使用速度维度
        self.chanel_num = chanel_num  # 获取通道数目
        datanum = len(all_file)
        train_offset = int(datanum * proportion[0])   # 以下三行为按照propotion划分各个部分的数据量
        test_offset = int(datanum * proportion[1]) + train_offset
        eval_offset = int(datanum * proportion[2]) + test_offset
        if split == 'train':
            print('Train Scene Data Loading..')
            # all_file = random.sample(range(len(all_file)),int(datanum * proportion[0]))
            # 随机采样
            all_file = all_file[:train_offset]
        if split == 'test':
            print('Test Scene Data Loading..')
            # 随机采样
            # all_file = random.sample(range(len(all_file)),int(datanum * proportion[1]))
            all_file = all_file[train_offset:test_offset]
        if split == 'eval':
            print('Eval Scene Data Loading..')
            # 随机采样
            # all_file = random.sample(range(len(all_file)),int(datanum * proportion[2]))
            all_file = all_file[test_offset:eval_offset]
        if split == 'whole':  # 使用当前目录下的全部数据
            print('Whole Scene Data Loading..')

        self.file_list = all_file
        self.file_len = len(all_file)
        # 只读取文件，不读取： 记录点数、初始化权重、标准化
        room_idxs = []
        # if resample == False:
        room_idxs = [i for i in range(len(all_file))]
            # 是否打乱room idx
            # room_idxs = random.sample(range(len(all_file)),len(all_file))
        # else:
        #     # resample 重采样操作
        #     num_all_point = []
        #     for file_name in all_file:
        #         path = os.path.join(self.carla_dir, file_name)
        #         if path[-3:] == 'npz':
        #             data = np.load(path, allow_pickle=True)['arr_0']
        #         elif path[-3:] == 'npy':
        #             data = np.load(path, allow_pickle=True)
        #         elif path[-3:] == 'txt':
        #             data = np.loadtxt(path)
        #         # data = np.load(path, allow_pickle=True)
        #         num_all_point.append(len(data))  # 记录点云数

            # sample_prob = num_all_point / np.sum(num_all_point)  # 单帧中包含的点云数占所有帧的数据点云数的比例
            # if numpoints == -1:
            #     num_iter = int(np.sum(num_all_point) * sample_rate / (8 * 1024))
            # else:
            #     num_iter = int(np.sum(num_all_point) * sample_rate / numpoints)
            # room_idxs = []
            # for index in range(self.file_len):
            #     room_idxs.extend([index] * int(round((sample_prob[index]) * num_iter)))  
            #     #  对点云数多的帧进行重采样（room_idx用于遍历所有数据，重采样可能后会重复采点云数较多的某一帧）
            #     #  例子：0,0,1,2,2,2……  在该例子中  1号帧点云数 < 0号帧点云数 < 3号帧点云数
            #     #  后续DataLoader遍历过程中会多次采样0号帧以及3号帧
        self.room_idxs = np.array(room_idxs)
        print("Totally {} samples in {} set.".format(len(self.room_idxs), split))

    def __getitem__(self, idx):
        room_idx = self.room_idxs[idx]
        roompath = os.path.join(self.carla_dir, self.file_list[room_idx])
        if roompath[-3:] == 'npz':
            raw_data = np.load(roompath)['arr_0']
        elif roompath[-3:] == 'npy':
            raw_data = np.load(roompath)
        elif roompath[-3:] == 'txt':
            raw_data = np.loadtxt(roompath)
        numpoints = self.numpoints
        
        if numpoints > len(raw_data):
            sample_points = raw_data
        else:
            # sample_points = farthest_point_sampling(raw_data, numpoints)
            sample_points = raw_data[:numpoints, :]

        point = sample_points[:, :4]
        label = sample_points[:, -1]
        point = np.asarray(point)
        label = np.asarray(label)
        N_points = len(label)
        
        

        return point, label

    def __len__(self):
        return len(self.room_idxs)


if __name__ == '__main__':
    point_data = CarlaDataset(carla_dir='data\\town_03_velocity_vector\\', split='eval', chanel_num=6,need_speed=False,resample=False,random_sample=False)
    train_loader = DataLoader(point_data, batch_size=16, shuffle=True, num_workers=0,
                              pin_memory=True, drop_last=True,
                              worker_init_fn=lambda x: np.random.seed(x + int(time.time())))

    # print('point data size:', point_data.__len__())
    # print('point data 0 shape:', point_data.__getitem__(0)[0].shape)
    # print('point label 0 shape:', point_data.__getitem__(0)[1].shape)
    raw_classes = ['Unlabeled', 'Building', 'Fence', 'Other', 'Pedestrian', 'Pole', 'RoadLine', 'Road',
               'SideWalk', 'Vegetation', 'Vehicles', 'Wall', 'TrafficSign', 'Sky', 'Ground', 'Bridge'
        , 'RailTrack', 'GuardRail', 'TrafficLight', 'Static', 'Dynamic', 'Water', 'Terrain']
    
    valid_label = [1, 4, 5, 7, 8, 9, 10, 11]
    trans_label = [0, 1, 2, 3, 4, 5, 6, 7]
    classes = [raw_classes[i] for i in valid_label]
    numclass = point_data.num_classes
    labelweights = np.zeros(numclass)
    class2label = {cls: i for i, cls in enumerate(classes)}
    seg_classes = class2label
    seg_label_to_cat = {}
    for i, cat in enumerate(seg_classes.keys()):
        seg_label_to_cat[i] = cat



    for i, (input, target) in enumerate(train_loader):
        print('calculating points distribution....', end = ' ')
        print(target.shape)
        print(input.shape)
        # batch_label = target.cpu().data.numpy()
        # tmp, _ = np.histogram(batch_label, range(numclass + 1))
        # labelweights += tmp

    labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))
    for l in range(numclass):
        print('class %s weight: %.3f' % (
            seg_label_to_cat[l] + ' ' * (numclass - len(seg_label_to_cat[l])), labelweights[l]))