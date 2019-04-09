import torch
import torch.nn as nn
import torch.nn.functional as F



class PCN1(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, dilation=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=2, stride=1)
        self.rotate = nn.Conv2d(128, 2, kernel_size=1, stride=1)
        self.cls_prob = nn.Conv2d(128, 2, kernel_size=1, stride=1)
        self.bbox = nn.Conv2d(128, 3, kernel_size=1, stride=1)

    def forward(self, x):
        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        x = F.relu(self.conv3(x), inplace=True)
        x = F.relu(self.conv4(x), inplace=True)
        cls_prob = F.softmax(self.cls_prob(x), dim=1)
        rotate = F.softmax(self.rotate(x), dim=1)
        bbox = self.bbox(x)
        return cls_prob, rotate, bbox

# caffe output for data shape
# data                        	 (1, 3, 24, 24)
# conv1_1                     	 (1, 16, 11, 11)
# conv2_1                     	 (1, 32, 5, 5)
# conv3_1                     	 (1, 64, 2, 2)
# fc4_1                       	 (1, 128, 1, 1)
# fc4_1_relu4_1_0_split_0     	 (1, 128, 1, 1)
# fc4_1_relu4_1_0_split_1     	 (1, 128, 1, 1)
# fc4_1_relu4_1_0_split_2     	 (1, 128, 1, 1)
# fc5_1                       	 (1, 2, 1, 1)
# cls_prob                    	 (1, 2, 1, 1)
# fc6_1                       	 (1, 2, 1, 1)
# rotate_cls_prob             	 (1, 2, 1, 1)
# bbox_reg_1                  	 (1, 3, 1, 1)

# caffe param
# conv1_1                     	 (16, 3, 3, 3) (16,)
# conv2_1                     	 (32, 16, 3, 3) (32,)
# conv3_1                     	 (64, 32, 3, 3) (64,)
# fc4_1                       	 (128, 64, 2, 2) (128,)
# fc5_1                       	 (2, 128, 1, 1) (2,)
# fc6_1                       	 (2, 128, 1, 1) (2,)
# bbox_reg_1                  	 (3, 128, 1, 1) (3,)

class PCN2(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 20, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(20, 40, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(40, 70, kernel_size=2, stride=1)
        self.fc = nn.Linear(70*3*3, 140)
        self.rotate = nn.Linear(140, 3)
        self.cls_prob = nn.Linear(140, 2)
        self.bbox = nn.Linear(140, 3)
        self.mp = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv1(x)
        x = F.pad(x, (0, 1, 0, 1))
        x = F.relu(self.mp(x), inplace=True)
        x = self.conv2(x)
        x = F.pad(x, (0, 1, 0, 1))
        x = F.relu(self.mp(x), inplace=True)
        x = F.relu(self.conv3(x), inplace=True)
        x = x.view(batch_size, -1)
        x = F.relu(self.fc(x), inplace=True)
        cls_prob = F.softmax(self.cls_prob(x), dim=1)
        rotate = F.softmax(self.rotate(x), dim=1)
        bbox = self.bbox(x)
        return cls_prob, rotate, bbox

# caffe output for data shape
# data                        	 (1, 3, 24, 24)
# conv1_2                     	 (1, 20, 22, 22)
# pool1_2                     	 (1, 20, 11, 11)
# conv2_2                     	 (1, 40, 9, 9)
# pool2_2                     	 (1, 40, 4, 4)
# conv3_2                     	 (1, 70, 3, 3)
# fc4_2                       	 (1, 140)
# fc4_2_relu4_2_0_split_0     	 (1, 140)
# fc4_2_relu4_2_0_split_1     	 (1, 140)
# fc4_2_relu4_2_0_split_2     	 (1, 140)
# fc5_2                       	 (1, 2)
# cls_prob                    	 (1, 2)
# fc6_2                       	 (1, 3)
# rotate_cls_prob             	 (1, 3)
# bbox_reg_2                  	 (1, 3)

# caffe param
# conv1_2                     	 (20, 3, 3, 3) (20,)
# conv2_2                     	 (40, 20, 3, 3) (40,)
# conv3_2                     	 (70, 40, 2, 2) (70,)
# fc4_2                       	 (140, 630) (140,)
# fc5_2                       	 (2, 140) (2,)
# fc6_2                       	 (3, 140) (3,)
# bbox_reg_2                  	 (3, 140) (3,)


class PCN3(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 24, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(24, 48, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(48, 96, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(96, 144, kernel_size=2, stride=1)
        self.fc = nn.Linear(144*3*3, 192)
        self.cls_prob = nn.Linear(192, 2)
        self.bbox = nn.Linear(192, 3)
        self.rotate = nn.Linear(192, 1)
        self.mp1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2)


    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv1(x)
        x = F.pad(x, (0, 1, 0, 1))
        x = F.relu(self.mp1(x), inplace=True)

        x = self.conv2(x)
        x = F.pad(x, (0, 1, 0, 1))
        x = F.relu(self.mp1(x), inplace=True)

        x = self.conv3(x)
        x = F.relu(self.mp2(x), inplace=True)
        x = F.relu(self.conv4(x), inplace=True)
        x = x.view(batch_size, -1)
        x = F.relu(self.fc(x), inplace=True)
        cls_prob = F.softmax(self.cls_prob(x), dim=1)
        rotate = self.rotate(x)
        bbox = self.bbox(x)
        return cls_prob, rotate, bbox

# caffe output for data shape
# data                        	 (1, 3, 48, 48)
# conv1_3                     	 (1, 24, 46, 46)
# pool1_3                     	 (1, 24, 23, 23)
# conv2_3                     	 (1, 48, 21, 21)
# pool2_3                     	 (1, 48, 10, 10)
# conv3_3                     	 (1, 96, 8, 8)
# pool3_3                     	 (1, 96, 4, 4)
# conv4_3                     	 (1, 144, 3, 3)
# fc5_3                       	 (1, 192)
# fc5_3_relu5_3_0_split_0     	 (1, 192)
# fc5_3_relu5_3_0_split_1     	 (1, 192)
# fc5_3_relu5_3_0_split_2     	 (1, 192)
# fc6_3                       	 (1, 2)
# cls_prob                    	 (1, 2)
# bbox_reg_3                  	 (1, 3)
# rotate_reg_3                	 (1, 1)


# caffe param
# conv1_3                     	 (24, 3, 3, 3) (24,)
# conv2_3                     	 (48, 24, 3, 3) (48,)
# conv3_3                     	 (96, 48, 3, 3) (96,)
# conv4_3                     	 (144, 96, 2, 2) (144,)
# fc5_3                       	 (192, 1296) (192,)
# fc6_3                       	 (2, 192) (2,)
# bbox_reg_3                  	 (3, 192) (3,)
# rotate_reg_3                	 (1, 192) (1,)

import os
from os.path import join as pjoin

def load_model():
    cwd = os.path.dirname(__file__)
    pcn1, pcn2, pcn3 = PCN1(), PCN2(), PCN3()
    pcn1.load_state_dict(torch.load(pjoin(cwd, 'pth/pcn1_sd.pth')))
    pcn2.load_state_dict(torch.load(pjoin(cwd, 'pth/pcn2_sd.pth')))
    pcn3.load_state_dict(torch.load(pjoin(cwd, 'pth/pcn3_sd.pth')))
    return pcn1, pcn2, pcn3