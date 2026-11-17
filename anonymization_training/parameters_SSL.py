import numpy as np

num_frames = 8
num_workers = 6
batch_size = 8  #16#24
learning_rate = 1e-4#1e-4#1e-5
num_epochs = 100
data_percentage = 1.0#.0
v_batch_size = 16 #48#80

fix_skip = 2
num_modes = 5
num_skips = 1
hflip = [0] #list(range(2))
cropping_fac1 = [0.8] #[0.7,0.85,0.8,0.75]

reso_h = 112
reso_w = 112

ori_reso_h = 240
ori_reso_w = 320

sr_ratio = 4


warmup_array = list(np.linspace(0.01,1, 5) + 1e-9)
warmup = len(warmup_array)

num_classes = 102#401
lr_patience = 0
scheduled_drop = 5

num_pa = 7 #Number of privacy attributes


ft_loss_weight = 1 #1 is equal weight with fb
fb_loss_weight = 0.2

learning_rate_fa = learning_rate
learning_rate_fb = 1e-5
learning_rate_ft = learning_rate

data_percentage_ucf101 = 1.0
batch_size_ucf101 = 24

data_percentage_vispr = 0.9578
batch_size_vispr = 8 # 64

#==========增加的参数===========
channel = 3 # 通道数
t_length = 5 # 视频序列总长度 4+1
fdim = 512 # 特征维度
mdim = 512 # 内存项维度
msize = 10 # 内存项数量
ped2_train_folder = '/opt/data/private/xwy/MNAD-master/dataset/ped2/training/frames' # ped2训练集路径
ped2_test_folder = '/opt/data/private/xwy/MNAD-master/dataset/ped2/testing/frames' # ped2测试集路径
avenue_train_folder = '/mnt/private/xwy/aed-mae/data/avenue/train/frames'
avenue_test_folder = '/mnt/private/xwy/aed-mae/data/avenue/test/frames'
avenue_labels_path = '/mnt/private/xwy/SPAct/datasets/frame_labels_avenue.npy'
shanghai_train_folder = '/mnt/private/xwy/aed-mae/data/shanghaitech_AED/train/frames'
shanghai_test_folder = '/mnt/private/xwy/aed-mae/data/shanghaitech_AED/test/frames'
shanghai_labels_path = '/mnt/private/xwy/SPAct/datasets/frame_labels_shanghai.npy'
height = 128 # 输入图像高度
width = 128 # 输入图像宽度
batch_size_ped2 = 8 # 批次大小
batch_size_avenue = 8 # 批次大小

loss_compact = 0.1 # 紧凑性损失
loss_separate = 0.1 # 分离性损失

vad_dataset = 'shanghai'
lambda_orth = 0.001