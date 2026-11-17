import numpy as np
from collections import OrderedDict
import os
import glob
import cv2
import torch.utils.data as data


rng = np.random.RandomState(2020) # 随机种子

# 读取图像并进行预处理:缩放至128*128并将像素归一化
def np_load_frame(filename, resize_height, resize_width):
    """
    加载图像路径并转换为numpy数组。注意颜色通道为BGR,颜色空间从[0, 255]归一化到[-1, 1]。

    :param filename: the full path of image 图像的完整路径
    :param resize_height: resized height 调整高度
    :param resize_width: resized width 调整宽度
    :return: numpy.ndarray 返回值
    """
    image_decoded = cv2.imread(filename) # 读取图像 BGR格式
    image_rgb = cv2.cvtColor(image_decoded, cv2.COLOR_BGR2RGB) # 转换为RGB
    image_resized = cv2.resize(image_decoded, (resize_width, resize_height)) # 调整尺寸 128*128
    image_resized = image_resized.astype(dtype=np.float32)  # 转换数据类型为float32
    image_resized = (image_resized / 127.5) - 1.0 # 归一化到[-1, 1] shaoe=[H,W,3]
    return image_resized



# 继承自DataLoader的自定义数据集类
class vadDataLoader(data.Dataset):
    def __init__(self, video_folder, transform, resize_height, resize_width, time_step=4, num_pred=1):
        self.dir = video_folder # 视频文件夹路径 存储视频帧的根目录（每个视频对应一个子目录，子目录下是连续帧的.png文件）
        self.transform = transform # 图像变换（如归一化、增强等） PyTorch 的Transform对象
        self.videos = OrderedDict() # 创建一个有序字典存储视频元信息
        self._resize_height = resize_height # 目标高度
        self._resize_width = resize_width # 目标宽度
        self._time_step = time_step  # 输入帧数（默认4帧）
        self._num_pred = num_pred # 预测帧数 默认1帧
        self.setup() # 初始化视频数据结构
        self.samples = self.get_all_samples() # 生成样本列表
        
     # 视频元信息加载   
    def setup(self):
        # 获取所有视频文件夹路径（如：/path/to/videos/video1, /path/to/videos/video2）
        videos = glob.glob(os.path.join(self.dir, '*'))
        for video in sorted(videos): # 遍历排序后的视频文件夹
            video_name = video.split('/')[-1] # 提取视频名（子目录名，如"video1"）
            self.videos[video_name] = {} # 初始化视频字典
            self.videos[video_name]['path'] = video # 视频子目录路径
            # 获取所有帧图像路径（*.png）并排序
            self.videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.png'))
            self.videos[video_name]['frame'].sort()
            self.videos[video_name]['length'] = len(self.videos[video_name]['frame'])  # 视频总帧数
            
    # 生成所有样本的起始帧路径列表       
    def get_all_samples(self):
        frames = [] # 存储所有样本的起始帧路径
        videos = glob.glob(os.path.join(self.dir, '*')) 
        for video in sorted(videos):
            video_name = video.split('/')[-1]

            # 遍历所有可能的起始帧（总帧数 - time_step，确保能取到连续time_step+num_pred帧）
            for i in range(len(self.videos[video_name]['frame'])-self._time_step):
                # 记录起始帧路径（如：/data/videos/video1/frame001.png）
                frames.append(self.videos[video_name]['frame'][i])
                           
        return frames #返回样本列表              
            
        
    def __getitem__(self, index):
        # 解析样本路径获取视频名和起始帧号
        # 示例：self.samples[0] = "/data/videos/video1/frame001.png"
        video_name = self.samples[index].split('/')[-2] # 视频名（子目录名
        frame_name = int(self.samples[index].split('/')[-1].split('.')[-2]) # 帧序号（如`0001.png`→1）
        
        batch = [] # 存储连续帧数据

        # 加载连续的time_step+num_pred帧
        for i in range(self._time_step+self._num_pred):
            # 读取第frame_name+i帧的图像（路径：self.videos[video_name]['frame'][frame_name+i]）
            image = np_load_frame(self.videos[video_name]['frame'][frame_name+i], self._resize_height, self._resize_width)
            # 应用变换（如转Tensor）
            if self.transform is not None:
                batch.append(self.transform(image)) 

        return np.concatenate(batch, axis=0) # 沿通道轴拼接（假设transform后为(C,H,W)，拼接后为(C*T,H,W)）
        
        
    def __len__(self):
        return len(self.samples)

