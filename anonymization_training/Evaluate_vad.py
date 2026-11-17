import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torch.nn.init as init
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.utils as v_utils
import matplotlib.pyplot as plt
import cv2
import math
from collections import OrderedDict
import copy
import time
from vad_utlis import vadDataLoader
from mnad_model.final_future_prediction_with_memory_spatial_sumonly_weight_ranking_top1 import *
from mnad_model.Reconstruction import *
from sklearn.metrics import roc_auc_score
from utils import *
import random
import glob
import io
import argparse

parser = argparse.ArgumentParser(description="MNAD")
parser.add_argument('--gpus', nargs='+', type=str, help='gpus')
parser.add_argument('--batch_size', type=int, default=4, help='batch size for training')
parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for test')
parser.add_argument('--h', type=int, default=128, help='height of input images')
parser.add_argument('--w', type=int, default=128, help='width of input images')
parser.add_argument('--c', type=int, default=3, help='channel of input images')
parser.add_argument('--method', type=str, default='pred', help='The target task for anoamly detection')
parser.add_argument('--t_length', type=int, default=5, help='length of the frame sequences')# 时间序列长度（连续帧数量）
parser.add_argument('--fdim', type=int, default=512, help='channel dimension of the features') # 特征维度
parser.add_argument('--mdim', type=int, default=512, help='channel dimension of the memory items') # 内存项维度
parser.add_argument('--msize', type=int, default=10, help='number of the memory items') # 内存项数量
parser.add_argument('--alpha', type=float, default=0.5, help='weight for the anomality score') # 异常分数加权系数
parser.add_argument('--th', type=float, default=0.01, help='threshold for test updating') # 测试时内存更新阈值
parser.add_argument('--num_workers', type=int, default=2, help='number of workers for the train loader')  # 训练数据加载线程数
parser.add_argument('--num_workers_test', type=int, default=1, help='number of workers for the test loader') # 测试数据加载线程数
parser.add_argument('--dataset_type', type=str, default='shanghai', help='type of dataset: ped2, avenue, shanghai') # 数据集类型
parser.add_argument('--dataset_path', type=str, default='/mnt/private/xwy/SPAct/anonymization_training/anonymized_avenue', help='directory of data') # 数据集路径
parser.add_argument('--model_dir', type=str, default='/mnt/private/xwy/SPAct/anonymization_training/saved_models/train0401/ft_model_21.pth', help='directory of model')  # 预训练模型路径
parser.add_argument('--m_items_dir', type=str, default='/mnt/private/xwy/SPAct/anonymization_training/saved_models/train0401/keys_21.pt', help='directory of model') # 内存项存储路径

args = parser.parse_args()

'''
# 配置可见的 GPU 设备，启用 CUDA 加速。
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # 设置CUDA设备顺序
if args.gpus is None:
    gpus = "0" # 默认使用GPU 0
    os.environ["CUDA_VISIBLE_DEVICES"]= gpus
else:
    gpus = ""
    for i in range(len(args.gpus)):
        gpus = gpus + args.gpus[i] + "," # 拼接GPU列表（如"0,1"）
    os.environ["CUDA_VISIBLE_DEVICES"]= gpus[:-1]  # 去除末尾逗号

torch.backends.cudnn.enabled = True # 启用cudnn加速（提升计算性能）
'''

test_folder = args.dataset_path+"/"+"/testing0401/frames" # 测试集路径

# 初始化
test_dataset = vadDataLoader(test_folder, transforms.Compose([
             transforms.ToTensor(), # 转换为张量           
             ]), resize_height=args.h, resize_width=args.w, time_step=args.t_length-1)

test_size = len(test_dataset) # 测试集样本数
# 测试集数据加载器（逐批加载数据）
test_batch = data.DataLoader(test_dataset, batch_size = args.test_batch_size, 
                             shuffle=False, num_workers=args.num_workers_test, drop_last=False)

# MSE损失
loss_func_mse = nn.MSELoss(reduction='none')
m_items_path = ''

with open(args.m_items_dir, 'rb') as f:
    m_items_buffer = io.BytesIO(f.read())
m_items = torch.load(m_items_buffer)

# 加载预训练的模型
model = torch.load(args.model_dir)
use_cuda = True
# 移动模型至GPU 
if torch.cuda.device_count()>1:
    print(f'Multiple GPUS found!')
    model=nn.DataParallel(model)
    model.cuda()

else:
    print('Only 1 GPU is available')
    model.cuda()

model.eval() # 模型切换为评估模式（禁用Dropout/BatchNorm）

labels = np.load('/mnt/private/xwy/SPAct/datasets/frame_labels_'+args.dataset_type+'.npy') # 加载帧级异常标签

videos = OrderedDict() # 有序字典存储视频信息 
videos_list = sorted(glob.glob(os.path.join(test_folder, '*'))) # 测试文件夹下的所有视频子目录（如视频1、视频2）
# 遍历每个视频，记录路径、帧列表和长度
for video in videos_list:
    video_name = video.split('/')[-1]  # 视频名（如01、02）
    videos[video_name] = {}
    videos[video_name]['path'] = video # 视频路径
    videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.png')) # 所有帧路径
    videos[video_name]['frame'].sort() # 按帧顺序排序
    videos[video_name]['length'] = len(videos[video_name]['frame']) # 视频总帧数

# 初始化标签列表、PSNR列表、特征距离列表
labels_list = []
label_length = 0 # 已处理的视频帧数
psnr_list = {} # 记录每个视频的PSNR（重建质量）
feature_distance_list = {}  # 记录每个视频的特征距离（内存紧凑性）

print('Evaluation of', args.dataset_type) # 打印当前评估的数据集

# Setting for video anomaly detection 初始化标签与对应关系
# 为每个视频分配对应的标签（根据视频长度截取标签）
for video in sorted(videos_list):
    video_name = video.split('/')[-1]
    if args.method == 'pred': # 预测模式（使用未来帧标签）
        # 标签偏移：前4帧无预测目标，从第5帧开始取标签
        labels_list = np.append(labels_list, labels[0][4+label_length:videos[video_name]['length']+label_length])
    else:
        labels_list = np.append(labels_list, labels[0][label_length:videos[video_name]['length']+label_length])
    label_length += videos[video_name]['length'] # 累计视频总帧数
    psnr_list[video_name] = [] # 初始化PSNR列表
    feature_distance_list[video_name] = []  # 初始化特征距离列表

label_length = 0 # 记录已处理的所有视频的总帧数
video_num = 0 # 当前处理的视频索引
label_length += videos[videos_list[video_num].split('/')[-1]]['length'] # 当前视频总帧数
m_items_test = m_items.clone() # 复制内存项（测试时动态更新）



for k,(imgs) in enumerate(test_batch): # 遍历测试批次（每批1个样本）
    # 更新当前视频索引（根据处理的帧数）
    if args.method == 'pred':
        if k == label_length-4*(video_num+1): # 预测模式：每处理完一个视频的（总帧数-4）帧后切换视频
            video_num += 1
            label_length += videos[videos_list[video_num].split('/')[-1]]['length']
    else:
        if k == label_length: # 重建模式 每处理完一个视频的总帧数后切换视频
            video_num += 1
            label_length += videos[videos_list[video_num].split('/')[-1]]['length']

    imgs = Variable(imgs).cuda() # 输入图像移至GPU（带梯度，但评估时不计算）
    
    # 模型前向传播（根据任务类型调用不同逻辑）
    if args.method == 'pred':
        # 预测模式：输入前4帧（3*4通道），预测第5帧
        outputs, feas, updated_feas, m_items_test, softmax_score_query, softmax_score_memory, _, _, _, compactness_loss = model.forward(imgs[:,0:3*4], m_items_test, False)
        # 计算重建图像与真实图像的MSE（归一化后）
        mse_imgs = torch.mean(loss_func_mse((outputs[0]+1)/2, (imgs[0,3*4:]+1)/2)).item()
        mse_feas = compactness_loss.item() # 特征紧凑损失（内存项与特征的距离）
 
        # Calculating the threshold for updating at the test time 计算点级异常分数（用于判断是否更新内存）
        point_sc = point_score(outputs, imgs[:,3*4:])
    
    else:
        # 重建模式 输入当前帧 直接重建
        outputs, feas, updated_feas, m_items_test, softmax_score_query, softmax_score_memory, compactness_loss = model.forward(imgs, m_items_test, False)
        mse_imgs = torch.mean(loss_func_mse((outputs[0]+1)/2, (imgs[0]+1)/2)).item()
        mse_feas = compactness_loss.item()

        # Calculating the threshold for updating at the test time
        point_sc = point_score(outputs, imgs)

    # 当分数低于阈值时，动态更新内存项
    if  point_sc < args.th:
        query = F.normalize(feas, dim=1) # 特征归一化
        query = query.permute(0,2,3,1) # # 调整维度（b, h, w, d）
        m_items_test = model.memory.update(query, m_items_test, False) # 更新内存项
    
    # 记录当前帧的PSNR和特征距离
    psnr_list[videos_list[video_num].split('/')[-1]].append(psnr(mse_imgs))
    feature_distance_list[videos_list[video_num].split('/')[-1]].append(mse_feas)


# 计算总异常分数（结合PSNR和特征距离）
anomaly_score_total_list = []
for video in sorted(videos_list):
    video_name = video.split('/')[-1]
    # 异常分数由两部分加权：PSNR的异常分数（越小越异常）和特征距离的异常分数（越大越异常）
    anomaly_score_total_list += score_sum(anomaly_score_list(psnr_list[video_name]),  # PSNR转换为异常分数（归一化）
                                     anomaly_score_list_inv(feature_distance_list[video_name]), args.alpha) # 特征距离转换为异常分数（逆归一化） # 加权系数

# 转换为numpy数组
anomaly_score_total_list = np.asarray(anomaly_score_total_list)
# 计算AUC（受试者工作特征曲线下面积）
accuracy = AUC(anomaly_score_total_list, np.expand_dims(1-labels_list, 0)) # 真实标签取反（标签1为异常）
# 输出结果
print('The result of ', args.dataset_type)
print('AUC: ', accuracy*100, '%')
