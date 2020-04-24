import numpy as np
import torch
import json
import random
import imageio
from torch import nn
from torch.nn import init
from collections import OrderedDict
from skimage import transform, feature
import cv2
import time

from HOG import *

label_to_idx = {
    'i2':0, 'i4':1, 'i5':2, 'io':3, 'ip':4, 
    'p11':5, 'p23':6, 'p26':7, 'p5':8, 'pl30':9, 
    'pl40':10, 'pl5':11, 'pl50':12, 'pl60':13, 'pl80':14, 
    'pn':15, 'pne':16, 'po':17, 'w57':18
}

idx_to_label = [
    'i2', 'i4', 'i5', 'io', 'ip', 'p11', 'p23', 'p26', 'p5', 'pl30', 
    'pl40', 'pl5', 'pl50', 'pl60', 'pl80', 'pn', 'pne', 'po', 'w57'
]

# 标准尺寸
width, height = (128, 128)

# 初始化训练数据
def init_train_data():
    start = time.time()
    with open("train.json", "r") as f:
    # "train-Copy1.json为手工精简版目录，仅用于验证程序正确性"
    #  with open("train-Copy1.json", "r") as f:
        dic = json.loads(f.read())
    names = list(dic)
    labels = list(dic.values())
    num_examples = len(names)
    r = random.random
    random.seed(0)
    random.shuffle(names, random = r)
    random.seed(0)
    random.shuffle(labels, random = r) # 随机打乱，用seed保证打乱顺序相同
    features = []
    idx_labels = []
    for i in range(num_examples):
        print(i)
        name = names[i]
        label = labels[i]
        path = "Train\\" + label + "\\" + name
        img = imageio.imread(path)
        new_img=transform.resize(img,(width,height))
        hog_img=feature.hog(new_img,orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
        block_norm='L2-Hys', visualize=False, transform_sqrt=True,
        feature_vector=False, multichannel=None)
        hog_img.resize((hog_img.shape[0],hog_img.shape[1],hog_img.shape[2]*hog_img.shape[3]*hog_img.shape[4]))
        features.append(hog_img)
        idx_labels.append(label_to_idx[label])
    print("time %.2f sce" % (time.time() - start))
    return features, idx_labels
# features 是一个 list，元素为np.array，形状为[宽 * 高 * 3（维度）]

# 读取训练数据
def train_data_iter(batch_size, features, labels):
    num_examples = len(features)
    for i in range(0, num_examples, batch_size):
        j = min(i + batch_size, num_examples)
        yield features[i: j], labels[i: j]

features, labels = init_train_data()

# 定义线性层
num_inputs, num_outputs = features[0].shape[0]*features[0].shape[1]*features[0].shape[2], 19
batch_size = 256

# 设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x):
        return x.view(x.shape[0], -1)

net = nn.Sequential(
    OrderedDict([
        ('flatten', FlattenLayer()),
        ('linear', nn.Linear(num_inputs, num_outputs))
    ])
)
init.normal_(net.linear.weight, mean = 0, std = 0.01)
init.constant_(net.linear.bias, val = 0)
# net.cuda()

# 损失函数
loss = nn.CrossEntropyLoss()

# 优化器
optimizer = torch.optim.SGD(net.parameters(), lr = 0.1)

# 迭代
epoch_num = 50
for epoch in range(epoch_num):
    start = time.time()
    train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
    for X, y in train_data_iter(batch_size, features, labels):
        
        # 获取数据，y为tensor，形状为[batch_size]
        y = torch.tensor(y, dtype = torch.long, device = device)
        
        # 获取数据，X为tensor，形状为[bath_size * 14 * 14 * 32(dim)]
        images = []
        for image in X:
            image = torch.tensor(image, dtype = torch.float, device = device)
            images.append(image)
        X = torch.stack(images)
        
        # 前向运算和损失
        y_hat = net(X)
        l = loss(y_hat, y).sum()

        # 梯度清零
        optimizer.zero_grad()

        # 后向梯度
        l.backward()
        optimizer.step()

        # 统计
        train_l_sum += l.item()
        train_acc_sum += (y_hat.argmax(dim = 1) == y).sum().item()
        n += y.shape[0]
    print('epoch %d, loss, %f, train acc %.3f, time %.2f sec' % (epoch + 1, train_l_sum / n, train_acc_sum / n, time.time() - start))

"""
说明：
自己写的HOG函数预处理实在太慢了...还是库里的比较好用
之后可以在服务器上跑一遍比较一下精确度

runout:

epoch 1, loss, 0.005912, train acc 0.632, time 1.38 sec
epoch 2, loss, 0.003169, train acc 0.848, time 1.16 sec
epoch 3, loss, 0.002400, train acc 0.888, time 1.15 sec
epoch 4, loss, 0.002013, train acc 0.907, time 1.16 sec
epoch 5, loss, 0.001774, train acc 0.917, time 1.19 sec
epoch 6, loss, 0.001607, train acc 0.923, time 1.19 sec
epoch 7, loss, 0.001484, train acc 0.928, time 1.18 sec
epoch 8, loss, 0.001387, train acc 0.932, time 1.40 sec
epoch 9, loss, 0.001308, train acc 0.936, time 1.51 sec
epoch 10, loss, 0.001242, train acc 0.938, time 1.32 sec
epoch 11, loss, 0.001186, train acc 0.941, time 1.37 sec
epoch 12, loss, 0.001137, train acc 0.943, time 1.41 sec
epoch 13, loss, 0.001094, train acc 0.946, time 1.38 sec
epoch 14, loss, 0.001055, train acc 0.947, time 1.38 sec
epoch 15, loss, 0.001021, train acc 0.949, time 1.40 sec
epoch 16, loss, 0.000990, train acc 0.951, time 1.37 sec
epoch 17, loss, 0.000961, train acc 0.953, time 1.44 sec
epoch 18, loss, 0.000934, train acc 0.954, time 1.42 sec
epoch 19, loss, 0.000910, train acc 0.955, time 1.37 sec
epoch 20, loss, 0.000887, train acc 0.956, time 1.38 sec
epoch 21, loss, 0.000866, train acc 0.957, time 1.37 sec
epoch 22, loss, 0.000846, train acc 0.959, time 1.36 sec
epoch 23, loss, 0.000827, train acc 0.959, time 1.42 sec
epoch 24, loss, 0.000810, train acc 0.960, time 1.39 sec
epoch 25, loss, 0.000793, train acc 0.961, time 1.36 sec
epoch 26, loss, 0.000777, train acc 0.962, time 1.36 sec
epoch 27, loss, 0.000762, train acc 0.963, time 1.35 sec
epoch 28, loss, 0.000748, train acc 0.964, time 1.37 sec
epoch 29, loss, 0.000734, train acc 0.964, time 1.37 sec
epoch 30, loss, 0.000721, train acc 0.965, time 1.33 sec
epoch 31, loss, 0.000708, train acc 0.965, time 1.36 sec
epoch 32, loss, 0.000696, train acc 0.965, time 1.37 sec
epoch 33, loss, 0.000685, train acc 0.966, time 1.34 sec
epoch 34, loss, 0.000674, train acc 0.967, time 1.37 sec
epoch 35, loss, 0.000663, train acc 0.968, time 1.35 sec
epoch 36, loss, 0.000653, train acc 0.968, time 1.30 sec
epoch 37, loss, 0.000643, train acc 0.969, time 1.39 sec
epoch 38, loss, 0.000633, train acc 0.970, time 1.37 sec
epoch 39, loss, 0.000624, train acc 0.970, time 1.37 sec
epoch 40, loss, 0.000614, train acc 0.971, time 1.35 sec
epoch 41, loss, 0.000606, train acc 0.971, time 1.39 sec
epoch 42, loss, 0.000597, train acc 0.972, time 1.35 sec
epoch 43, loss, 0.000589, train acc 0.972, time 1.34 sec
epoch 44, loss, 0.000581, train acc 0.973, time 1.35 sec
epoch 45, loss, 0.000573, train acc 0.973, time 1.37 sec
epoch 46, loss, 0.000566, train acc 0.974, time 1.33 sec
epoch 47, loss, 0.000558, train acc 0.974, time 1.37 sec
epoch 48, loss, 0.000551, train acc 0.975, time 1.34 sec
epoch 49, loss, 0.000544, train acc 0.975, time 1.34 sec
epoch 50, loss, 0.000537, train acc 0.975, time 1.35 sec
"""