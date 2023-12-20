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
# Point4D with smaller MLP(speed chanel)

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

            
            self.conv1_speed = torch.nn.Conv1d(1, 128, 1)
            self.conv2_speed = torch.nn.Conv1d(128, self.k, 1)
            self.bn1_speed = nn.BatchNorm1d(64)
        self.feat = PointNetEncoder(global_feat=False, feature_transform=True, channel=channel)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128 + self.k, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        x_speed = x[:,-1,:]
        x = x[:, :3, :]
        # B, C, N = x.shape
        # B C N
        # 16 1 8192
        # s = pd.Series(x_speed.reshape(-1).cpu())
        # s = s.kurt()
        # ratio = np.exp(-1 * s / 3)
        ratio = 0.3
        # 4 2 1
        # print("ratio :%f" %ratio)
        x_speed = x_speed[:, np.newaxis, :]
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        # x -- B * 3 * N
        
        x_speed_ = F.relu(self.conv1_speed(x_speed)) # 速度特征 B * 128 * N
        x_speed = F.relu(self.conv2_speed(x_speed_)) # 速度pred( B T N )-- T 语义标签数
        x_speed_exp = x_speed_ # B * 128 * N
        x_speed_ = torch.max(x_speed_, 2, keepdim=True)[0] # 速度特征 B * 128 * 1
        
        x, trans, trans_feat = self.feat(x) # B * 1088 * N
        # 在高维空间融合速度特征
        # x = torch.cat((x, x_speed), 1) # 节点1
        x = F.relu(self.bn1(self.conv1(x))) # B * 512 * N
        x = F.relu(self.bn2(self.conv2(x))) # B * 256 * N
        x = F.relu(self.bn3(self.conv3(x))) # B * 128 * N
        x_cord = x # 空间特征 B * 128 * N
        x_cord_prime = x_cord[torch.randperm(x_cord.size(0))] # B * 128 * N
        # 在低维空间做速度特征融合
        x = torch.cat((x, x_speed), 1) # 节点2
        x = self.conv4(x) # B * C * N 
        x = x.transpose(2, 1).contiguous() # B * N * C
        x = F.log_softmax(x.view(-1, self.k), dim=-1)
        x = x.view(batchsize, n_pts, self.k)

        x_cord_global = F.adaptive_max_pool1d(x_cord, 1).view(batchsize, -1) # B * 128
        x_cord_global_prime = x_cord_global[torch.randperm(x_cord_global.size(0))] # B * 128 * N
        
        

        # x_cord_local x_cord_global
        # x_speed_local x_speed_global

        result = {
            'x' : x, # B, N, K
            'trans_feat' : trans_feat,
            'x_cord' : x_cord, # B * 128 * N
            'x_cord_prime' : x_cord_prime, # B * 128 * N
            'x_cord_global_prime' : x_cord_global_prime, # B * 128
            'x_cord_global' : x_cord_global, # B * 128
            'x_speed' : x_speed_.squeeze(), # B * 128
            'x_speed_exp' : x_speed_exp # B * 128 * N
        }
        
        return result


class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, x_cord, x_cord_prime,
                x_cord_global, x_cord_global_prime, 
                x_speed, x_speed_exp):
        
        loss = F.nll_loss(pred, target)
        MMI = DeepMILoss().to('cuda:0')
        lLoss, gLoss, MILoss = MMI(x_cord_global, x_cord_global_prime, 
                            x_cord, x_cord_prime,
                            x_speed, x_speed_exp)


        total_loss = loss + MILoss
        return total_loss

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


class PointNetEncoderNew(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, channel=3):
        super(PointNetEncoderNew, self).__init__()
        self.stn = STN3d(channel)
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 64, 1)
        self.conv3 = torch.nn.Conv1d(64, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        B, D, N = x.size() # (batch_size, 3, num_points)
        trans = self.stn(x) # (batch_size, 3, 3)
        x = x.transpose(2, 1)
        if D >3 :
            x, feature = x.split(3,dim=2)
        x = torch.bmm(x, trans)
        if D > 3:
            x = torch.cat([x,feature],dim=2)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x))) # (batch_size, 64, n)

        ########################################################
        x = F.relu(self.bn2(self.conv2(x))) # (batch_size, 64, n)
        x_local = x # (b, 64, n)
        ########################################################

        if self.feature_transform:
            trans_feat = self.fstn(x)  # (batch_size, 64, 64)
            x = x.transpose(2, 1) 
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1) # (batch_size, 64, n)
        else:
            trans_feat = None

        pointfeat = x # (batch_size, 64, n)
        x = F.relu(self.bn3(self.conv3(x))) # (batch_size, 128, n)
        x = self.bn4(self.conv4(x)) # (batch_size, 1024, n)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024) # (batch_size, 1024)
        if self.global_feat:
            return x, trans, trans_feat, x_local
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N) # (batch_size, 1024, n)
            return torch.cat([x, pointfeat], 1), trans, trans_feat, x_local
            ## #1 global_feature: (batch_size, 1024)
            ## #1 else: (batch_size, 1088, n)
            #2 -- (batch_size, 3, 3)
            #3 -- (batch_size, 64, n)

class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, channel=3):
        super(PointNetEncoder, self).__init__()
        self.stn = STN3d(channel)
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
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
        x = x.transpose(2, 1)
        if D > 3:
            feature = x[:, :, 3:]
            x = x[:, :, :3]
            x = torch.cat([x, feature], dim=2)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
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

class GlobalinfolossNet(nn.Module):
    def __init__(self):
        super(GlobalinfolossNet, self).__init__()
        self.c1 = nn.Conv1d(256, 256, kernel_size=1, bias=False)
        self.c2 = nn.Conv1d(256, 64, kernel_size=1, bias=False)
        self.c3 = nn.Conv1d(64, 32, kernel_size=1, bias=False)
        self.l0 = nn.Linear(32, 1)
    
    def forward(self, x_global, c):
        # input size: (b, 64)
        # x_global = b*64   c = b*64
        xx = torch.cat((x_global, c), dim = 1)  # -> (b, 128)
        h = xx.unsqueeze(dim=2) # -> (b, 128, 1)
        h = F.relu(self.c1(h)) # -> (b, 128, 1)
        h = F.relu(self.c2(h)) # -> (b, 64, 1)
        h = F.relu(self.c3(h)) # -> (b, 32, 1)
        h = h.view(h.shape[0], -1) # (b, 32)

        return self.l0(h)  # b*1


## repeat shape code before computing the loss
## input local feature b*128*1024 ( batch_size * features * num_points )
## input repeated shape code b*128*1024
## each pair xx:  b*(128+128)*1024
class LocalinfolossNet(nn.Module):
    def __init__(self):
        super(LocalinfolossNet, self).__init__()
        self.conv1 = nn.Conv1d(256, 128, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(128, 1, kernel_size=1, bias=False)
    
    def forward(self, x_local, c):
        # x_local: b* 64* n
        # c : b* 64* n
        xx = torch.cat((x_local, c), dim=1) # -> (b, 128, num_points)
        h = F.relu(self.conv1(xx))  # (b, 128, num_points) -> (b, 64, num_points)
        h = F.relu(self.conv2(h)) #(b, 64, num_points) -> (b, 64, num_points)
        h = F.relu(self.conv3(h))  # (b, 64, num_points) -> (b, 1, num_points)
        h = h.view(h.shape[0], -1) # (b, num_points)
        return h # (b, num_points)


class DeepMILoss(nn.Module):
    def __init__(self):
        super(DeepMILoss, self).__init__()

        self.global_d = GlobalinfolossNet()
        self.local_d = LocalinfolossNet()
        
   
    def forward(self, x_global, x_global_prime, x_local, x_local_prime, c, c_p):
        # x_local: (batch_size, 64, num_points) 
        # x_local_prime: (batch_size, 64, num_points)
        # x_global: (batch_size, 64)  
        # x_global_prime: (batch_size, 64)
        # c: (batch_size, 64, num_points) --- c3
        # c_p: (batch_size, 64) --- c2

        ###### local loss ############
        Ej = -F.softplus(-self.local_d(c, x_local)).mean() # positive pairs
        Em = F.softplus(self.local_d(c, x_local_prime)).mean() # negetive pairs
        LOCAL = (Em - Ej) * 0.5
        

        ###### global loss ###########
        Ej = -F.softplus(-self.global_d(c_p, x_global)).mean() # positive pairs
        Em = F.softplus(self.global_d(c_p, x_global_prime)).mean() # negetive pairs
        GLOBAL = (Em - Ej) * 0.5

        ######### combine local and global loss ###############
        ToT = LOCAL + GLOBAL

        return LOCAL, GLOBAL, ToT # tensor, a value