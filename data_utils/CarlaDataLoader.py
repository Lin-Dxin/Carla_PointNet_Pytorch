import os
import numpy as np
import torch
# from plyfile import PlyData
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from time import time
import tqdm


class CarlaDataset(Dataset):
    # label_weights = np.random.normal(size=5)

    def __init__(self, carla_dir, transform=None, split='train', proportion=[0.7, 0.2, 0.1],
                 num_classes=5, sample_rate=0.1, numpoints=1024 * 8, need_speed=True, chanel_num = 3,
                 block_size=1.0, resample=False,random_sample=True, normalized=False):

        
        self.split = split  # 区分训练集或者测试集（当数据按文件划分后，可以用whole读取所有数据
        self.proportion = proportion  # 数据划分比例
        self.random_sample = random_sample
        # rootpath = os.path.abspath('..')
        self.num_classes = num_classes  # 语义类别数
        self.block_size = block_size  # 用于重采样的重采样块大小
        self.carla_dir = os.path.join(carla_dir)  # 数据路径
        self.transform = transform # 用于数据强化，主要是旋转、裁剪数据（目前尚未使用
        self.label_weights = np.random.normal(size=num_classes)  # 用于记录数据分布（指各个语义标签的数据占总体数据的比例）
        self.numpoints = numpoints  # 单帧中采样的点数
        all_file = os.listdir(self.carla_dir)  # 用于记录数据量
        self.need_speed = need_speed  # 用于区分是否使用速度维度
        self.chanel_num = chanel_num  # 获取通道数目
        datanum = len(all_file)
        self.normalized = normalized
        self.chanel_num = chanel_num
        train_offset = int(datanum * proportion[0])   # 以下三行为按照propotion划分各个部分的数据量
        test_offset = int(datanum * proportion[1]) + train_offset
        eval_offset = int(datanum * proportion[2]) + test_offset
        if split == 'train':
            print('Train Scene Data Loading..')
            all_file = all_file[:train_offset]
        if split == 'test':
            print('Test Scene Data Loading..')
            all_file = all_file[train_offset:test_offset]
        if split == 'eval':
            print('Eval Scene Data Loading..')
            all_file = all_file[test_offset:eval_offset]
        if split == 'whole':  # 使用当前目录下的全部数据
            print('Whole Scene Data Loading..')

        self.file_list = all_file
        self.file_len = len(all_file)
        # 只读取文件，不保存： 记录点数、初始化权重、标准化
        room_idxs = []
        if resample == False:
            room_idxs = [i for i in range(len(all_file))]
        else:
            # resample 重采样操作
            num_all_point = []
            for file_name in all_file:
                path = os.path.join(self.carla_dir, file_name)
                if path[-3:] == 'npz':
                    data = np.load(path, allow_pickle=True)['arr_0']
                elif path[-3:] == 'npy':
                    data = np.load(path, allow_pickle=True)
                elif path[-3:] == 'txt':
                    data = np.loadtxt(path)
                # data = np.load(path, allow_pickle=True)
                num_all_point.append(len(data))  # 记录点云数

            sample_prob = num_all_point / np.sum(num_all_point)  # 单帧中包含的点云数占所有帧的数据点云数的比例
            num_iter = int(np.sum(num_all_point) * sample_rate / numpoints)  
            room_idxs = []
            for index in range(self.file_len):
                room_idxs.extend([index] * int(round((sample_prob[index]) * num_iter)))  
                #  对点云数多的帧进行重采样（room_idx用于遍历所有数据，重采样可能后会重复采点云数较多的某一帧）
                #  例子：0,0,1,2,2,2……  在该例子中  1号帧点云数 < 0号帧点云数 < 3号帧点云数
                #  后续DataLoader遍历过程中会多次采样0号帧以及3号帧
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
        point = []
        label = []

        # for _raw in raw_data:
        #     if self.need_speed:
        #         temp = [_raw[0], _raw[1], _raw[2], _raw[3]]
        #     else:
        #         temp = [_raw[0], _raw[1], _raw[2]]
        #     point.append(temp)
        #     label.append(_raw[5])
        # point = np.asarray(point)
        # label = np.asarray(label)
        if self.normalized:
            xyz = raw_data[:, :3]
            xyz_max = np.max(xyz,axis=0)
            xyz_min = np.min(xyz,axis=0)
            normalize_xyz = (xyz[:]-xyz_min) / (xyz_max - xyz_min)
            point = np.concatenate((xyz,normalize_xyz),axis=1)
        else:
            point = raw_data[:, :self.chanel_num]
        label = raw_data[:, -1]

        point = np.asarray(point)
        label = np.asarray(label)
        N_points = len(label)
        cnt = 0
        if self.random_sample:

            while True:
                # 对一帧内的点进行随机采样
                # 随机选择三个点，并在三个点的周围划定区域，Block_Size决定了区域大小
                # 最终输出数据为Block内的点云
                center = point[np.random.choice(N_points)][:3]
                block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0]
                block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0]
                point_idxs = np.where(
                    (point[:, 0] >= block_min[0]) & (point[:, 0] <= block_max[0]) & (point[:, 1] >= block_min[1]) & (
                            point[:, 1] <= block_max[1]))[0]
                # print(point_idxs.size)
                cnt += 1
                # print(cnt)
                if point_idxs.size > 1024 or cnt > 100:
                    # print("success! with cnt:")
                    # print(cnt)
                    break
        else:
            point_idxs = np.array([i for i in range(len(label))])
        if point_idxs.size >= self.numpoints:
            selected_point_idxs = np.random.choice(point_idxs, self.numpoints, replace=False)
        else:
            selected_point_idxs = np.random.choice(point_idxs, self.numpoints, replace=True)
        selected_data = np.array(point, dtype=np.float32)[selected_point_idxs]
        selected_label = np.array(label, dtype=np.float32)[selected_point_idxs]
        return selected_data, selected_label

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