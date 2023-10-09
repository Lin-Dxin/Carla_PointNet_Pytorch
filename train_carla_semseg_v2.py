from data_utils.CarlaDataLoader import CarlaDataset
from torch.utils.data import DataLoader
import numpy as np
import time

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



if Model == "pointnet2":
    from models.pointnet2_semseg_carla import get_model, get_loss
elif Model == "pointnet":
    from models.pointnet_semseg_carla import get_model, get_loss
elif Model == "pointnet3":
    from models.point4dnet import get_model, get_loss
elif Model == "pointnet3.1":
    from models.point4dnet_v2 import get_model, get_loss
elif Model == "pointnet3.3":
    from models.point4dnet_v3 import get_model, get_loss
elif Model == "pointnet4":
    from models.point4d_multi_scale import get_model, get_loss
elif Model == "pointnet4.3":
    from models.point4d_multi_scale_3 import get_model, get_loss


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
    if TSB_RECORD:
        # dummy_input = torch.randn(16,3,8192)
        # dummy_input = dummy_input.to(device)
        # log_writer.add_graph(classifier, dummy_input)
        log_string('initial learning rate: %f' % learning_rate)
        log_string('decay_rate: %f' % decay_rate)
    for epoch in range(state_epoch, epoch_num):
        log_string('**** Epoch %d  ****' % (epoch + 1))
        start_time = time.time()
        lr = max(learning_rate * (decay_rate ** (epoch // step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        num_batches = len(train_loader)
        total_correct = 0
        total_seen = 0
        # kurt_sum = 0
        loss_sum = 0
        classifier = classifier.train()
        
        for i, (points, target) in tqdm(enumerate(train_loader), total=len(train_loader), smoothing=0.9):
            # print(points.shape)
            # print(target)
            optimizer.zero_grad()
            # print(target)
            points = points.data.numpy()
            points = torch.Tensor(points)
            points, target = points.float().to(device), target.long().to(device)
            points = points.transpose(2, 1)

            seg_pred, trans_feat, x_cord, x_speed = classifier(points)
            seg_pred = seg_pred.to(device)
            # trans_feat = trans_feat.to(device)
            seg_pred = seg_pred.contiguous().view(-1, numclass)
            x_cord = x_cord.contiguous().view(-1, numclass)
            x_speed = x_speed.contiguous().view(-1, numclass)
            batch_label = target.view(-1, 1)[:, 0].cpu().data.numpy()
            target = target.view(-1, 1)[:, 0]
            loss = criterion(seg_pred, x_cord, x_speed, target ,trans_feat, weights)
            loss.backward()
            optimizer.step()

            pred_choice = seg_pred.cpu().data.max(1)[1].numpy()
            correct = np.sum(pred_choice == batch_label)
            total_correct += correct
            # total_seen += (16 * 4096)
            total_seen += points.shape[0] * points.shape[2]
            loss_sum += loss
            # kurt_sum += kurt
            # break
        if TSB_RECORD:
            log_writer.add_scalar('Loss/train', float(loss_sum / num_batches), epoch)
            log_writer.add_scalar('ACC/train', total_correct / float(total_seen), epoch)
            # log_writer.add_scalar('Kurt/train', float(kurt_sum / num_batches), epoch)
        log_string('Training mean loss: %f' % (loss_sum / num_batches))
        log_string('Training accuracy: %f' % (total_correct / float(total_seen)))
        # log_string('Training mean kurt: %f' % (kurt_sum / num_batches))
        train_acc.append((total_correct / float(total_seen)))
        train_loss.append((loss_sum / num_batches))
        with torch.no_grad():
            num_batches = len(test_loader)
            total_correct = 0
            total_seen = 0
            loss_sum = 0
            # kurt_sum = 0
            labelweights = np.zeros(numclass)
            total_seen_class = [0 for _ in range(numclass)]
            total_correct_class = [0 for _ in range(numclass)]
            total_iou_deno_class = [0 for _ in range(numclass)]
            temp_class_acc = [0 for _ in range(numclass)]
            classifier = classifier.eval()
            print('---- EPOCH %03d EVALUATION ----' % (epoch + 1))
            for i, (points, target) in tqdm(enumerate(test_loader), total=len(test_loader), smoothing=0.9):
                points = points.data.numpy()
                points = torch.Tensor(points)
                points, target = points.float().to(device), target.long().to(device)
                points = points.transpose(2, 1)

                seg_pred, trans_feat, x_cord, x_speed = classifier(points)
                # seg_pred = seg_pred.to(device)
                # trans_feat = trans_feat.to(device)
                # seg_pred [16,8192,8]
                pred_val = seg_pred.contiguous().cpu().data.numpy()
                seg_pred = seg_pred.contiguous().view(-1, numclass)
                x_cord = x_cord.contiguous().view(-1, numclass)
                x_speed = x_speed.contiguous().view(-1, numclass)
                
                # seg_pred = seg_pred.contiguous().view(-1, numclass)
                batch_label = target.cpu().data.numpy()
                target = target.view(-1, 1)[:, 0]
                loss = criterion(seg_pred, x_cord, x_speed, target ,trans_feat, weights)
                # kurt_sum += kurt
                loss_sum += loss
                pred_val = np.argmax(pred_val, 2)
                correct = np.sum((pred_val == batch_label))
                total_correct += correct
                total_seen += points.shape[0] * points.shape[2]
                # print(np.histogram(batch_label, range(24)))
                tmp, _ = np.histogram(batch_label, range(numclass + 1))
                labelweights += tmp

                for l in range(0, numclass):
                    total_seen_class[l] += np.sum((batch_label == l))
                    total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l))
                    total_iou_deno_class[l] += np.sum(((pred_val == l) | (batch_label == l)))
                    
                    

                # break
        labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))
        mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=float) + 1e-6))
        for l in range(0,numclass):
            if total_seen_class[l] == 0:
                temp_class_acc[l] = 0.01
            else:
                temp_class_acc[l] = total_correct_class[l] / total_seen_class[l]
        
        if TSB_RECORD:
            log_writer.add_scalar('mIoU/eval', mIoU, epoch)
            log_writer.add_scalar('loss/eval', (loss_sum / float(num_batches)), epoch)
            # log_writer.add_scalar('Kurt/eval', float(kurt_sum / num_batches), epoch)
            log_writer.add_scalar('acc/eval', (total_correct / float(total_seen)), epoch)
        

        log_string('eval mean loss: %f' % (loss_sum / float(num_batches)))
        log_string('eval point avg class IoU: %f' % mIoU)
        log_string('eval point accuracy: %f' % (total_correct / float(total_seen)))
        # log_string('eval mean kurt: %f' % (kurt_sum / num_batches))
        class_acc.append(temp_class_acc)
        validate_acc.append((total_correct / float(total_seen)))
        validate_miou.append(mIoU)
        validate_loss.append((loss_sum / float(num_batches)))
        # log_string('eval point avg class acc: %f' % (
        #     np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=float) + 1e-6))))
        iou_per_class_str = '------- IoU --------\n'
        for l in range(numclass):
            iou_per_class_str += 'class %s weight: %.3f' % (
                seg_label_to_cat[l] + ' ' * (numclass - len(seg_label_to_cat[l])), labelweights[l])
            if total_iou_deno_class[l] != 0:
                iou_per_class_str += ', IoU: %.3f ' % (total_correct_class[l] / float(total_iou_deno_class[l]))
            else:
                iou_per_class_str += ', IoU: UnValid '
            iou_per_class_str += ', class_acc: %.3f \n' % temp_class_acc[l]
        log_string(iou_per_class_str)
        end_time = time.time()
        spd_time = end_time - start_time
        log_string('Spending Time: %f' % spd_time)
        log_string('Eval mean loss: %f' % (loss_sum / num_batches))
        log_string('Eval accuracy: %f' % (total_correct / float(total_seen)))
        logger.info('Save model...')
        savepath = str(checkpoints_dir) + '/state_dict.pth'
        log_string('Saving at %s' % savepath)
        state = {
                'epoch': epoch,
                'class_avg_iou': mIoU,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
        torch.save(state, savepath)
        log_string('Saving model....')
        if mIoU >= best_iou:
            best_iou = mIoU
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/best_model.pth'
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'class_avg_iou': mIoU,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saving model....')
        log_string('Best mIoU: %f' % best_iou)
    if K_FOLD:
        eval_save_path = str(checkpoints_dir) + '/evaluation_data%s'% partition_str + '.pth'
    else:
        eval_save_path = str(checkpoints_dir) + '/evaluation_data.pth'
    log_string('Save evaluation data at %s' %eval_save_path)
    evaluation = {
        'train_acc' : train_acc,
        'train_loss' : train_loss,
        'validate_acc' : validate_acc,
        'validate_loss' : validate_loss,
        'validate_miou' : validate_miou,
        'class_acc' : class_acc
    }
    torch.save(evaluation, eval_save_path)
