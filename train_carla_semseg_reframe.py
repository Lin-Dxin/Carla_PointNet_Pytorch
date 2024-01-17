from data_utils.CarlaDataLoader import CarlaDataset
from torch.utils.data import DataLoader
import numpy as np
import time
import importlib
import torch
from tqdm import tqdm
import datetime
import os
import sys
import logging
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import json

with open('semseg_config.json') as f:
  json_data = json.load(f)

# 读取semseg.json的配置文件

TRANS_LABEL = json_data['TRANS_LABEL'] # 是否使用原标签
_carla_dir = json_data['_carla_dir'] # 若不使用Kflod则该目录为主
NEED_SPEED = json_data['NEED_SPEED'] # 是否使用4D数据
TSB_RECORD = json_data['TSB_RECORD'] # 是否使用Tensorboard记录实验过程
Model = json_data['Model'] # 使用的模型 pointnet / pointnet2
epoch_num = json_data['epoch_num'] # 设定epoch数
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_path = json_data['model_path']  # 需要一个初始化模型
K_FOLD = json_data['K_FOLD'] # 是否使用KFLOD训练
SAVE_INIT = json_data['SAVE_INIT'] # 将这个选项设为True、Load_Init设为False 可以在log/checkpoint/初始化生成一个initial_state.pth的初始化模型
LOAD_INIT = json_data['LOAD_INIT']  # 不能与Save_Init相同
DATA_RESAMPLE = json_data['DATA_RESAMPLE'] # 是否重采样数据
RANDOM_RESAMPLE = json_data['RANDOM_RESAMPLE'] # 是否随机采样
CHANEL_NUM = json_data['CHANEL_NUM']

if K_FOLD:
    partition = json_data['partition'] # 0 - 9
    partition_str = str(partition)
    train_data_dir_pre = json_data['train_data_dir_pre']
    train_data_dir = train_data_dir_pre + partition_str+ '/train'  
    # 需要有TrainAndValidateData_0、TrainAndValidateData_1 …… TrainAndValidateData_9 十个文件夹存放各个分布的数据
    validate_data_dir_pre = json_data['validate_data_dir_pre']
    validate_data_dir = validate_data_dir_pre + partition_str+ '/validate'
    # model_info需要自己修改成对应的实验标题
    model_info_pre = json_data['model_info_pre']
    model_info =  model_info_pre + partition_str  # 最好能区分是否4D数据、使用pn或者pn++ 例：3D_pn2_part
    # 不用自行添加partition，已经记录下来了

Model = 'models/'+json_data['model']
get_model, get_loss = importlib.import_module(Model)


if TRANS_LABEL:
    raw_classes = ['Unlabeled', 'Building', 'Fence', 'Other', 'Pedestrian', 'Pole', 'RoadLine', 'Road',
                   'SideWalk', 'Vegetation', 'Vehicles', 'Wall', 'TrafficSign', 'Sky', 'Ground', 'Bridge'
        , 'RailTrack', 'GuardRail', 'TrafficLight', 'Static', 'Dynamic', 'Water', 'Terrain']
    # raw_classes = np.array(raw_classes)
    valid_label = [1, 4, 5, 7, 8, 9, 10, 11]
    trans_label = [0, 1, 2, 3, 4, 5, 6, 7]
    classes = [raw_classes[i] for i in valid_label]
    # classes = ['Building', 'Road', 'Sidewalk', 'Vehicles', 'Wall']  # 最终标签列表
    # print(classes)
    numclass = len(valid_label)
else:
    classes = ['Unlabeled', 'Building', 'Fence', 'Other', 'Pedestrian', 'Pole', 'RoadLine', 'Road',
               'SideWalk', 'Vegetation', 'Vehicles', 'Wall', 'TrafficSign', 'Sky', 'Ground', 'Bridge'
        , 'RailTrack', 'GuardRail', 'TrafficLight', 'Static', 'Dynamic', 'Water', 'Terrain']
    numclass = 23
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True





if __name__ == '__main__':

    PROPOTION = [0.7, 0.2, 0.1]
    # prepare for log file
    
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('sem_seg')
    experiment_dir.mkdir(exist_ok=True)
    if K_FOLD:
        experiment_dir = experiment_dir.joinpath(model_info)
    else:
        experiment_dir = experiment_dir.joinpath(timestr)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if NEED_SPEED:
        file_handler = logging.FileHandler('%s/logs/4d_train.txt' % experiment_dir)
    else:
        file_handler = logging.FileHandler('%s/logs/3d_train.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    if TSB_RECORD:
        log_writer = SummaryWriter('%s/logs/' % experiment_dir)


    def log_string(str):
        logger.info(str)
        print(str)


    # train
    # config dataloader

    if K_FOLD:
        train_dataset = CarlaDataset(split='whole', carla_dir=train_data_dir, chanel_num=CHANEL_NUM,num_classes=numclass, need_speed=NEED_SPEED, proportion=PROPOTION,resample=DATA_RESAMPLE)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0,
                                pin_memory=True, drop_last=True, chanel_num=CHANEL_NUM)
        test_dataset = CarlaDataset(split='whole', carla_dir=validate_data_dir, chanel_num=CHANEL_NUM, num_classes=numclass, need_speed=NEED_SPEED, proportion=PROPOTION,resample=DATA_RESAMPLE)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=0,
                                pin_memory=True, drop_last=True, chanel_num=CHANEL_NUM)
    else:
        train_dataset = CarlaDataset(split='train', carla_dir=_carla_dir, num_classes=numclass, chanel_num=CHANEL_NUM, need_speed=NEED_SPEED, proportion=PROPOTION,resample=DATA_RESAMPLE)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0,
                                pin_memory=True, drop_last=True)
        test_dataset = CarlaDataset(split='test', carla_dir=_carla_dir, num_classes=numclass, chanel_num=CHANEL_NUM, need_speed=NEED_SPEED, proportion=PROPOTION,resample=DATA_RESAMPLE)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=0,
                                pin_memory=True, drop_last=True)
    # print(train_dataset.__len__())
    # print(test_dataset.__len__())
    log_string("Using Model:%s" % Model)
    log_string("Using 4D data:%s" % NEED_SPEED)
    log_string("The number of training data is: %d" % len(train_dataset))
    log_string("The number of test data is: %d" % len(test_dataset))

        
    classifier = get_model(numclass, need_speed=NEED_SPEED,chanel_num=CHANEL_NUM).to(device)  # loading model\
    log_string(json_data)
    if LOAD_INIT:
        checkpoint = torch.load(model_path,map_location = device)
        classifier.load_state_dict(checkpoint['model_state_dict'])
        if checkpoint.__contains__('epoch'):
            state_epoch = checkpoint['epoch']
        else:
            state_epoch = 0
    else:
        state_epoch = 0
        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                torch.nn.init.xavier_normal_(m.weight.data)
                torch.nn.init.constant_(m.bias.data, 0.0)
            elif classname.find('Linear') != -1:
                torch.nn.init.xavier_normal_(m.weight.data)
                torch.nn.init.constant_(m.bias.data, 0.0)

        classifier = classifier.apply(weights_init)
    # save initial model
    if SAVE_INIT is True:
        initial_state = {
                'model_state_dict': classifier.state_dict(),
                'epoch':0
            }
        init_savepath = str(checkpoints_dir) + '/initial_state.pth'
        torch.save(initial_state, init_savepath)
        log_string('Saving model at %s' %init_savepath)
        log_string('Shuting down')
        sys.exit()
        
    classifier.to(device)
    criterion = get_loss().to(device)  # loss function
    classifier.apply(inplace_relu)
    learning_rate = 0.001
    decay_rate = 0.0001
    step_size = 10
    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = step_size
    temp = np.random.normal(size=numclass)
    weights = torch.Tensor(temp).to(device)
    
    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=decay_rate
    )

    if LOAD_INIT:
        if state_epoch != 0:
            optimizer = optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum


    
    best_iou = 0

    train_acc = []
    train_loss = []
    validate_acc = []
    validate_loss = []
    validate_miou = []
    class_acc = []
    
    
