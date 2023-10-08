#
# from ast import main
# from tkinter.tix import MAIN
import torch.nn.parallel
import torch.utils.data

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import pandas as pd


class get_model(nn.Module):
    def __init__(self, num_class, need_speed = True, chanel_num = 3):
        super(get_model, self).__init__()
        self.k = num_class
        channel = chanel_num
        self.need_speed = need_speed
        if need_speed:
            # self.feat_speed = PointNetEncoder(global_feat=False, feature_transform=True, channel=1)
            # self.conv1_speed = torch.nn.Conv1d(1088, 512, 1)
            # self.conv2_speed = torch.nn.Conv1d(512, 256, 1)
            # self.conv3_speed = torch.nn.Conv1d(256, 128, 1)
            # self.conv4_speed = torch.nn.Conv1d(128, self.k, 1)
            # self.bn1_speed = nn.BatchNorm1d(512)
            # self.bn2_speed = nn.BatchNorm1d(256)
            # self.bn3_speed = nn.BatchNorm1d(128)
            # B N x
            # x1 16,64,8192
            # x2 16,1024,8192
            # x3 16,1088,8192
            # x4 16,8192,8


            self.conv0_speed = torch.nn.Conv1d(1,64,1)
            self.conv1_speed = torch.nn.Conv1d(64, 1024, 1)
            self.conv2_speed = torch.nn.Conv1d(1024, 1088, 1)
            self.conv3_speed = torch.nn.Conv1d(1088, 8, 1)
            self.conv4_speed = torch.nn.Conv1d(8, self.k, 1)
            self.bn0_speed = nn.BatchNorm1d(64)
            self.bn1_speed = nn.BatchNorm1d(1024)
            self.bn2_speed = nn.BatchNorm1d(1088)
            self.bn3_speed = nn.BatchNorm1d(8)
        # self.feat = PointNetEncoder(global_feat=False, feature_transform=True, channel=channel) # 1088
        # pointnet Encoder
        self.stn = STN3d(channel)
        self.conv1_pn = torch.nn.Conv1d(channel, 64, 1)
        self.conv2_pn = torch.nn.Conv1d(64, 128, 1)
        self.conv3_pn = torch.nn.Conv1d(128, 1024, 1)
        self.bn1_pn = nn.BatchNorm1d(64)
        self.bn2_pn = nn.BatchNorm1d(128)
        self.bn3_pn = nn.BatchNorm1d(1024)
        self.global_feat = False
        self.feature_transform = False
        
        
        # PointNet MLP_Layers
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

        # Combined Fearture Layers
        self.conv1_cb = torch.nn.Conv1d(64, self.k, 1) # stage1
        self.conv2_cb = torch.nn.Conv1d(1024, self.k, 1) # stage2
        self.conv3_cb = torch.nn.Conv1d(1088, self.k, 1) # stage3
        # self.conv4_cb = torch.nn.Conv1d(128, self.k, 1) # stage4

    def forward(self, x):
        x_speed = x[:,-1,:]
        x = x[:, :3, :]
        # 16 1 8192
        s = pd.Series(x_speed.reshape(-1).cpu())
        s = s.kurt()
        # ratio = np.exp(-1 * s / 3)
        ratio = 0.1
        # 4 2 1
        # print("ratio :%f" %ratio)
        x_speed = x_speed[:, np.newaxis, :]
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        # x -- B * 3 * N
        # x, trans, trans_feat = self.feat(x) # B * 1088 * N

        # PointNet
        B, D, N = x.size()
        trans = self.stn(x)
        # x = x.transpose(2, 1)
        x = F.relu(self.bn1_pn(self.conv1_pn(x))) # stage_1
        x_stageOne = x
        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2_pn(self.conv2_pn(x))) 
        x = self.bn3_pn(self.conv3_pn(x)) # stage_2
        x_stageTwo = x
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat  is False:
            # segmentation
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            x = torch.cat([x, pointfeat], 1) # stage_3
            x_stageThree = x

        # MLP 
        x = F.relu(self.bn1(self.conv1(x))) # B * 512 * N
        x = F.relu(self.bn2(self.conv2(x))) # B * 256 * N
        x = F.relu(self.bn3(self.conv3(x))) # B * 128 * N
        x = self.conv4(x) # B * C * N 
        x = x.transpose(2, 1).contiguous() # B * N * C
        x = F.log_softmax(x.view(-1, self.k), dim=-1)
        x = x.view(batchsize, n_pts, self.k) # stage_4
        x_stageFour = x
        

        # Speed_MLP
        if self.need_speed:
            # B * 1 * N
            # x_speed, trans_speed, trans_feat_speed = self.feat_speed(x_speed) 
            x_speed_one = F.relu(self.bn0_speed(self.conv0_speed(x_speed))) # stage_1
            x_speed_two = F.relu(self.bn1_speed(self.conv1_speed(x_speed_one))) # stage_2
            x_speed_three = F.relu(self.bn2_speed(self.conv2_speed(x_speed_two))) # stage_3
            x_speed_four = F.relu(self.bn3_speed(self.conv3_speed(x_speed_three))) # stage_4
            x_speed = self.conv4_speed(x_speed_four) # full connect
            x_speed = x_speed.transpose(2, 1).contiguous()
            x_speed = F.log_softmax(x_speed.view(-1, self.k), dim=-1)
            x_speed = x_speed.view(batchsize, n_pts, self.k)
            
            # x = x + 0.3 * x_speed
            
            trans_feat = trans_feat
        # B x N
        # x1 16,64,8192
        # x2 16,1024,8192
        # x3 16,1088,8192
        # x4 16,8192,8
        # v B x N
        CF_One = x_stageOne + ratio * x_speed_one
        CF_One = self.conv1_cb(CF_One)
        CF_One = CF_One.transpose(2,1).contiguous()
        CF_One = F.log_softmax(CF_One.view(-1, self.k), dim=-1)
        CF_One = CF_One.view(batchsize, n_pts, self.k)


        CF_Two = x_stageTwo + ratio * x_speed_two
        CF_Two = self.conv2_cb(CF_Two)
        CF_Two = CF_Two.transpose(2,1).contiguous()
        CF_Two = F.log_softmax(CF_Two.view(-1, self.k), dim=-1)
        CF_Two = CF_One.view(batchsize, n_pts, self.k)


        CF_Three = x_stageThree + ratio * x_speed_three
        CF_Three = self.conv3_cb(CF_Three)
        CF_Three = CF_Three.transpose(2,1).contiguous()
        CF_Three = F.log_softmax(CF_Three.view(-1, self.k), dim=-1)
        CF_Three = CF_One.view(batchsize, n_pts, self.k)


        x_stageFour = x_stageFour.transpose(2,1)
        CF_Four = x_stageFour + ratio * x_speed_four
        CF_Four = CF_Four.transpose(2,1).contiguous()
        CF_Four = F.log_softmax(CF_Four.view(-1, self.k), dim=-1)
        CF_Four = CF_One.view(batchsize, n_pts, self.k)

        x = x + (CF_One + CF_Two + CF_Three + CF_Four + x_speed)* ratio 
        return x, trans_feat, s


class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat, weight):
        loss = F.nll_loss(pred, target, weight=None)
        # mat_diff_loss = feature_transform_reguliarzer(trans_feat)
        # total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return loss

class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.to(x.device)
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.to(x.device)
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, channel=3):
        super(PointNetEncoder, self).__init__()
        self.stn = STN3d(channel)
        self.conv1 = torch.nn.Conv1d(channel, 64, 1) # stage_1
        self.conv2 = torch.nn.Conv1d(64, 128, 1) # stage_2
        self.conv3 = torch.nn.Conv1d(128, 1024, 1) # stage_3
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        B, D, N = x.size()
        trans = self.stn(x)
        # x = x.transpose(2, 1)
        # if D > 3:
        #     feature = x[:, :, 3:]
        #     x = x[:, :, :3]
        # if D == 1:

        # x = torch.bmm(x, trans)
        # if D > 3:
        #     x = torch.cat([x, feature], dim=2)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x))) # stage_1

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x 
        x = F.relu(self.bn2(self.conv2(x))) # stage_2
        x = self.bn3(self.conv3(x)) # stage_3
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


def feature_transform_reguliarzer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.to(trans.device)
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss


