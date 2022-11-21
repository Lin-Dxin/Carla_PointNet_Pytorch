from data_utils.CarlaDataLoader import CarlaDataset
from torch.utils.data import DataLoader
import numpy as np
import time
# from models.pointnet_semseg_carla import get_model, get_loss
import torch
from tqdm import tqdm
import datetime
import os
import sys
import logging
from pathlib import Path

TRANS_LABEL = True
NEED_SPEED = False
CARLA_DIR = './data/carla_scene_01/TestData/'
Model = 'pointnet'
model_path = './3dbest_model.pth'
if Model == "pointnet2":
    from models.pointnet2_semseg_carla import get_model, get_loss
else:
    from models.pointnet_semseg_carla import get_model, get_loss

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

if __name__ == '__main__':

    PROPOTION = [0.7, 0.2, 0.1]
    # prepare for log file
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('sem_seg_test')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath(timestr)
    experiment_dir.mkdir(exist_ok=True)
    # checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    # checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if NEED_SPEED:
        file_handler = logging.FileHandler('%s/logs/4d_eval_logs.txt' % experiment_dir)
    else:
        file_handler = logging.FileHandler('%s/logs/3d_eval_logs.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


    def log_string(str):
        logger.info(str)
        print(str)


    # config dataset and data Loader
    dataset = CarlaDataset(split='whole', carla_dir=CARLA_DIR, need_speed=NEED_SPEED, proportion=PROPOTION)
    dataLoader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0,
                            pin_memory=True, drop_last=True)
    log_string("The number of test data is: %d" % len(dataset))

    # load model
    # numclass = 5
    classifier = get_model(numclass, need_speed=NEED_SPEED).to(device)  # loading model
    criterion = get_loss().to(device)  # loss function
    
    checkpoint = torch.load(model_path, map_location=device)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier = classifier.eval()

    start_time = time.time()
    with torch.no_grad():
        num_batches = len(dataLoader)
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        labelweights = np.zeros(numclass)
        total_seen_class = [0 for _ in range(numclass)]
        total_correct_class = [0 for _ in range(numclass)]
        total_iou_deno_class = [0 for _ in range(numclass)]
        for i, (points, target) in tqdm(enumerate(dataLoader), total=len(dataLoader), smoothing=0.9):
            it_start_time = time.time()
            points = points.data.numpy()
            points = torch.Tensor(points)
            points, target = points.float().to(device), target.long().to(device)
            points = points.transpose(2, 1)
            seg_pred, trans_feat = classifier(points)
            pred_val = seg_pred.contiguous().cpu().data.numpy()
            seg_pred = seg_pred.contiguous().view(-1, numclass)
            batch_label = target.cpu().data.numpy()
            target = target.view(-1, 1)[:, 0]
            pred_val = np.argmax(pred_val, 2)
            it_end_time = time.time()
            # print('\niteration time:', (it_end_time - it_start_time), '\n')
            correct = np.sum((pred_val == batch_label))
            total_correct += correct
            total_seen += points.shape[0] * points.shape[2]
            tmp, _ = np.histogram(batch_label, range(numclass + 1))
            labelweights += tmp

            for l in range(0, numclass):
                total_seen_class[l] += np.sum((batch_label == l))
                total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l))
                total_iou_deno_class[l] += np.sum(((pred_val == l) | (batch_label == l)))
        labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))
        # sum = 0
        # valid = 0
        # for l in range(len(total_correct_class)):
        #     if (total_iou_deno_class != 0):
        #         valid = valid + 1
        #         sum += np.array(total_correct_class[l]) / (np.array(total_iou_deno_class[l], dtype=float) + 1e-6)
        # mIoU = sum / valid
        mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=float) + 1e-6))
        # log_string('eval mean loss: %f' % (loss_sum / float(num_batches)))
        log_string('eval point avg class IoU: %f' % mIoU)
        log_string('eval point accuracy: %f' % (total_correct / float(total_seen)))
        iou_per_class_str = '------- IoU --------\n'
        for l in range(numclass):
            iou_per_class_str += 'class %s weight: %f' % (
                seg_label_to_cat[l] + ' ' * (numclass - len(seg_label_to_cat[l])), labelweights[l])
            if total_iou_deno_class[l] != 0:
                iou_per_class_str += ', IoU: %f \n' % (total_correct_class[l] / float(total_iou_deno_class[l]))
            else:
                iou_per_class_str += ', IoU: UnValid\n'
        log_string(iou_per_class_str)
        end_time = time.time()
        spd_time = end_time - start_time
        log_string('Spending Time: %f' % spd_time)
        log_string('Eval accuracy: %f' % (total_correct / float(total_seen)))
        log_string('Eval mIoU: %f' % mIoU)
