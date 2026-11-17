import numpy as np
# 各种参数
num_frames = 16  # 每个视频抽取的帧数
num_workers = 10  # 多线程的进程数
batch_size = 32  #16#24  # 每个批次的样本数量
learning_rate = 1e-3  #1e-4#1e-5  # 学习率
num_epochs = 300  # 迭代轮数
data_percentage = 1.0#.0  # 使用数据的百分比
v_batch_size = 48  #80

fix_skip = 2
num_modes = 5 # ?
num_skips = 1
hflip = [0] #list(range(2)) 是否进行水平翻转的数据增强
cropping_fac1 = [0.8] #[0.7,0.85,0.8,0.75]  随机裁剪的比例

reso_h = 112  # 图像的高度
reso_w = 112  # 图像的高度

ori_reso_h = 240  # 原始图像的高度
ori_reso_w = 320  # 原始图像的宽度

sr_ratio = 4  # 超分辨率重建的比例因子


warmup_array = list(np.linspace(0.01,1, 5) + 1e-9)
warmup = len(warmup_array)

num_classes = 102  # 401 分类任务中的类别数量
lr_patience = 0
scheduled_drop = 5

# 参数文件