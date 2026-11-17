# This dataloader returns images from UCF101 and VISPR for the reconstrunction training for UNet, which will be used as a initialization of any minimax training.
# 此数据加载器从UCF101和VISPR返回图像，用于UNet的重构训练，该训练将用作任何极小极大训练的初始化。
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision

from torchvision import transforms, utils
import config as cfg
import random, glob, traceback
import pickle
import parameters_BL as params
import json
import math
import cv2
from tqdm import tqdm
import time
import torchvision.transforms as trans
from torchvision.utils import save_image

#定义重构的类
class reconstruction_dataloader(Dataset):

    def __init__(self, datasplit, shuffle = True, ucf101_precentage= 0.01, data_percentage = 1.0):
        # self.labeled_datapaths = open(os.path.join(cfg.path_folder,'10percentTrain_crcv.txt'),'r').read().splitlines()
        self.datasplit = datasplit

        if self.datasplit == 'train':
            self.vispr_datapath = os.path.join(cfg.vispr_path, 'train2017')  #cfg是config.py
            self.ucf101_datapath = os.path.join(cfg.ucf101_path, 'train')

            self.vispr = glob.glob(self.vispr_datapath + '/*.jpg')  #搜索当前目录下所有的jpg文件
            self.ucf101 = open('ucf101_training_frames.txt','r',encoding='utf-16').read().splitlines()
            
        
        if self.datasplit == 'test':
            self.vispr_datapath = os.path.join(cfg.vispr_path, 'test2017')
            self.ucf101_datapath = os.path.join(cfg.ucf101_path, 'test')

            self.vispr = glob.glob(self.vispr_datapath + '/*.jpg')
            self.ucf101 = open('ucf101_testing_frames.txt','r',encoding='utf-16').read().splitlines()
        random.shuffle(self.ucf101)  # 打乱原序列（直接改变序列

        #限制数据量
        self.ucf101_precentage = ucf101_precentage  
        self.ucf101_limit = int(len(self.ucf101)*self.ucf101_precentage)
        self.ucf101 = self.ucf101[0: self.ucf101_limit]

        self.all_paths = self.vispr + self.ucf101  # 合并VISPR和抽样后的UCF101
            
        if shuffle:  # 打乱顺序的意义是确保模型不会记住特定顺序的样本
            random.shuffle(self.all_paths)

        self.data_percentage = data_percentage
        self.data_limit = int(len(self.all_paths)*self.data_percentage)
        self.data = self.all_paths[0: self.data_limit]
        self.PIL = trans.ToPILImage()  # 初始化一个PIL格式的图像，以便于应用支持处理PIL的函数
        self.TENSOR = trans.ToTensor()  #将PIL图像转换成PyTorch张量（该张量适合于神经网络处理。包含归一化和CHW（通道高度宽度）两个操作
        self.erase_size = 11
    
    # 返回数据集长度
    def __len__(self):  
        return len(self.data)

    #       
    def __getitem__(self,index):        
        clip, img_path = self.process_data(index)
        return clip, img_path
    
    def process_data(self, idx):
    
        # label_building
        img_path = self.data[idx]
       
        # img_building
        img = self.build_image(img_path)

        return img, img_path
    
    def build_image(self, img_path):

        try:
            img = torchvision.io.read_image(img_path)  # 读取图像文件并返回一个PyTorch张量
            if img.shape[0] == 1:
                # print(img.shape)
                img = img.repeat(3, 1, 1)
            if not img.shape[0]==3:
                # print(f'{img_path} has {img.shape[0]} channels')
                return None
            # print(img.shape)
            # exit()
            ori_reso_w = img.shape[-1]  # wide
            ori_reso_h = img.shape[1]  # height

            #（输入给augmentation的参数）根据随机参数进行多种数据增强操作，以模拟不同光照和视角条件下的图像，增强模型的泛化能力
            random_array = np.random.rand(2,8)
            x_erase = np.random.randint(0,params.reso_h, size = (2,))
            y_erase = np.random.randint(0,params.reso_w, size = (2,))

            #裁剪
            cropping_factor1 = np.random.uniform(0.6, 1, size = (2,)) # 随即裁剪因子 on an average cropping factor is 80% i.e. covers 64% area
            x0 = np.random.randint(0, ori_reso_w - ori_reso_w*cropping_factor1[0] + 1)  # 裁剪起始点
            y0 = np.random.randint(0, ori_reso_h - ori_reso_h*cropping_factor1[0] + 1)  

            contrast_factor1 = np.random.uniform(0.9,1.1, size = (2,)) # 随机调整图像对比度
            hue_factor1 = np.random.uniform(-0.05,0.05, size = (2,))  # 随机调整图像色调
            saturation_factor1 = np.random.uniform(0.9,1.1, size = (2,))  # 随机调整图像饱和度
            brightness_factor1 = np.random.uniform(0.9,1.1,size = (2,))  # 随机调整图像亮度
            gamma1 = np.random.uniform(0.85,1.15, size = (2,))  # 随机伽马校正

            # 随机擦除
            erase_size1 = np.random.randint(int(self.erase_size/2),self.erase_size, size = (2,))
            erase_size2 = np.random.randint(int(self.erase_size/2),self.erase_size, size = (2,))
            random_color_dropped = np.random.randint(0,3,(2)) #在擦除区域中随机丢弃的颜色通道
            
            #对训练数据随机进行数据增强，对测试数据进行中心裁剪和调整大小，使其尺寸符合要求
            if self.datasplit == 'train':
                img = self.augmentation(img, random_array[0], x_erase[0], y_erase[0], cropping_factor1[0], x0, y0, contrast_factor1[0], hue_factor1[0], saturation_factor1[0], brightness_factor1[0],                           gamma1[0],erase_size1[0],erase_size2[0], random_color_dropped[0])/255.0
            elif self.datasplit == 'test':
                img = self.test_augmentation(img)/255.0
            
            try:
                assert(len(img.shape)!=0)
                
                return img
            except:
                # print(frames_full)
                print(f'1Image {img_path} Failed')
                # print(traceback.print_exc())
                return None   

        except:
            print(f'2Image {img_path} Failed')
            # print(traceback.print_exc())

            return None
    
    # 实现增强操作
    def augmentation(self, image, random_array, x_erase, y_erase, cropping_factor1,\
        x0, y0, contrast_factor1, hue_factor1, saturation_factor1, brightness_factor1,\
        gamma1,erase_size1,erase_size2, random_color_dropped):
        # ori_reso_h = image.shape[]
        ori_reso_h,ori_reso_w = image.shape[1], image.shape[-1]

        image = trans.functional.resized_crop(image,y0,x0,int(ori_reso_h*cropping_factor1),int(ori_reso_w*cropping_factor1),(params.reso_h,params.reso_w))


        if random_array[0] < 0.125/2:
            image = trans.functional.adjust_contrast(image, contrast_factor = contrast_factor1) #0.75 to 1.25
        if random_array[1] < 0.3/2 :
            image = trans.functional.adjust_hue(image, hue_factor = hue_factor1) # hue factor will be between [-0.25, 0.25]*0.4 = [-0.1, 0.1]
        if random_array[2] < 0.3/2 :
            image = trans.functional.adjust_saturation(image, saturation_factor = saturation_factor1) # brightness factor will be between [0.75, 1,25]
        if random_array[3] < 0.3/2 :
            image = trans.functional.adjust_brightness(image, brightness_factor = brightness_factor1) # brightness factor will be between [0.75, 1,25]
        if random_array[0] > 0.125/2 and random_array[0] < 0.25/2:
            image = trans.functional.adjust_contrast(image, contrast_factor = contrast_factor1) #0.75 to 1.25
        if random_array[4] > 0.9:
            image = trans.functional.rgb_to_grayscale(image, num_output_channels = 3)
            if random_array[5] > 0.25:
                image = trans.functional.adjust_gamma(image, gamma = gamma1, gain=1) #gamma range [0.8, 1.2]
        if random_array[6] > 0.5:
            image = trans.functional.hflip(image)

        # image = trans.functional.to_tensor(image)

        if random_array[6] < 0.5/2 :
            image = trans.functional.erase(image, x_erase, y_erase, erase_size1, erase_size2, v=0) 

        return image

    # 中心裁剪和调整大小
    def test_augmentation(self, image):
        h,w = image.shape[1], image.shape[-1]
        side = min(h,w)
        image = trans.functional.center_crop(image, side)
        image = trans.functional.resize(image,(params.reso_h,params.reso_w))
        return image

# 用于整理训练数据批次，将图像数据堆叠为张量，并返回图像张量和对应的路径列表，忽略None值的数据。            
def collate_fn_train(batch):  # batch 批次 

    f_clip, vid_path = [], []
    # print(len(batch))
    for item in batch:
        if not (item[0] == None or item[1] == None):
            f_clip.append(item[0]) # I might need to convert this tensor to torch.float
            vid_path.append(item[1])
        # else:
            # print('oh no2')
    # print(len(f_clip))
    f_clip = torch.stack(f_clip, dim=0)
    
    return f_clip, vid_path 

if __name__ == '__main__':

    train_dataset = reconstruction_dataloader(datasplit = 'train', shuffle = False, data_percentage = 1.0)  # 实例化训练数据集，指定数据集分割为train，不打乱顺序，使用全部数据。
    train_dataloader = DataLoader(train_dataset, batch_size=params.batch_size, \
        shuffle=True, num_workers=params.num_workers, collate_fn=collate_fn_train)
    # 【数据集】 【每个批次中的样本数量】 【打乱顺序】 【指定用于数据加载的子进程数量】【将多个样本合并成一个批次】

    print(f'Step involved: {len(train_dataset)/params.batch_size}')  # 打印加载数据的步骤数
    t=time.time()
    visualize = True
    dl_vispath = cfg.dl_vispath + '/'+ str(int(time.time()))
    
    # 可视化 用于检查数据和预处理结果
    for i, (clip, vid_path) in enumerate(train_dataloader):
        if i%10 == 0:
            print()
            # clip = clip.permute(0,1,3,4,2)
            print(f'Full_clip shape is {clip.shape}')
            if visualize:
                if not os.path.exists(dl_vispath):
                    os.makedirs(dl_vispath)
                save_image(clip, dl_vispath + '/dl_visualization.png', padding=5, nrow=int(clip.shape[0]/5))
                pickle.dump(clip, open(dl_vispath + '/f_clip.pkl','wb'))
                break
            # pickle.dump(label, open('label.pkl','wb'))
            # exit()
    print(f'Time taken to load data is {time.time()-t}')


