"""
Jet Substructure Models used in the LogicNets paper
"""

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import ReLU


class modellab1(nn.Module):
    def __init__(self, info):
        super(modellab1, self).__init__()
        # 1 输入图像通道, 6 输出通道, 5x5 卷积核
        self.conv1 = nn.Conv2d(3, 6, 5)
        # 6 输入图像通道, 16 输出通道, 5x5 卷积核
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        # 这里论文上写的是conv,官方教程用了线性层
        self.fc1 = nn.Linear(16 * 5 * 5, 120) 
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 最大池化，2*2的窗口滑动
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # 如果滑动窗口是方形，可以直接写一个数
        #把所有特征展平，num_flat_features(x)==x.size()[1:].numel()
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # 获取出了batch的其它维度
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def get_modellab1(info):
    return modellab1(info)