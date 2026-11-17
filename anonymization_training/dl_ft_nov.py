import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms, utils
import config as cfg
import random
import pickle
import parameters_BL as params
import json
import math
import cv2
# from tqdm import tqdm
import time
import torchvision.transforms as trans
# from decord import VideoReader

class baseline_dataloader_train_strong(Dataset):

    def __init__(self, shuffle = True, data_percentage = 1.0, split = 1):

        if split == 1:
            self.all_paths = open(os.path.join(cfg.path_folder, 'UCF-101/ucfTrainTestlist/trainlist01.txt'),'r').read().splitlines()
        elif split ==2: 
            self.all_paths = open(os.path.join(cfg.path_folder, 'UCF-101/ucfTrainTestlist/trainlist02.txt'),'r').read().splitlines()
        elif split ==3: 
            self.all_paths = open(os.path.join(cfg.path_folder, 'UCF-101/ucfTrainTestlist/trainlist03.txt'),'r').read().splitlines()
        else:
            print(f'Invalid split input: {split}')
        
        self.classes= json.load(open(cfg.class_mapping_ucf101))
        self.shuffle = shuffle

        if self.shuffle:
            random.shuffle(self.all_paths)
        
        self.data_percentage = data_percentage
        self.data_limit = int(len(self.all_paths)*self.data_percentage)
        self.data = self.all_paths[0: self.data_limit]
        self.PIL = trans.ToPILImage()
        self.TENSOR = trans.ToTensor()
        self.erase_size = 19



    def __len__(self):
        return len(self.data)
            
    def __getitem__(self,index):        
        clip, label, vid_path = self.process_data(index)
        return clip, label, vid_path

    def process_data(self, idx):
    
        # label_building
#         vid_path = self.data[idx]
        vid_path = cfg.path_folder + 'ucf101/videos/' + self.data[idx].split(' ')[0] #取第idx个视频的路径
        # print(vid_path)
        # exit()
        label = self.classes[vid_path.split('/')[-2]]
#         label = self.classes[vid_path.split('/')[6]] # THIS MIGHT BE DIFFERNT AFTER STEVE MOVE THE PATHS  
        
        # clip_building
        clip = self.build_clip(vid_path)

        return clip, label, vid_path

    def build_clip(self, vid_path):

        try:
            # 尝试打开视频文件
            cap = cv2.VideoCapture(vid_path)
            # 设置视频读取的起始帧为第0帧
            cap.set(1, 0)
             # 获取视频的总帧数
            frame_count = cap.get(7)

            ############################# frame_list maker start here#################################
            # 固定每帧之间的跳过帧数，从参数配置中获取
            skip_frames_full = params.fix_skip #frame_count/(params.num_frames)
            # 计算按照固定跳过帧数取帧后，剩余的帧数
            left_over = frame_count - params.fix_skip*params.num_frames

            # start_frame_full = np.random.randint(0,int(left_over)) # here 0 can be replaced with random starting point, but, for now it's getting complicated
            # 如果剩余帧数小于等于0，起始帧设为0
            if left_over <=0:
                start_frame_full = 0
            # 否则，随机选择一个起始帧
            else:    
                start_frame_full = np.random.randint(0,int(left_over))

            # 生成要提取的帧的索引列表
            frames_full = start_frame_full + np.asarray([int(int(skip_frames_full)*f) for f in range(params.num_frames)])

            # print(frames_full)
            # exit()
            # 如果最后一帧的索引超过了视频的总帧数，将最后一帧的索引设为总帧数减1
            if frames_full[-1] >= frame_count:
                # print('some corner case not covered')
                frames_full[-1] = int(frame_count-1)
            ################################ frame list maker finishes here ###########################

            ################################ actual clip builder starts here ##########################  
            full_clip = [] # 用于存储提取的帧
            list_full = [] # 用于存储实际提取的帧的索引
            count = -1 # 当前帧的计数器
            random_array = np.random.rand(2,8) # 生成随机数组，用于数据增强的随机决策
            x_erase = np.random.randint(0,params.reso_h, size = (2,)) # 随机生成擦除区域的x坐标
            y_erase = np.random.randint(0,params.reso_w, size = (2,)) # 随机生成擦除区域的y坐标

            # 随机生成裁剪因子，用于随机裁剪
            cropping_factor1 = np.random.uniform(0.6, 1, size = (2,)) # on an average cropping factor is 80% i.e. covers 64% area
            x0 = np.random.randint(0, params.ori_reso_w - params.ori_reso_w*cropping_factor1[0] + 1) # 随机生成裁剪的起始x坐标
            y0 = np.random.randint(0, params.ori_reso_h - params.ori_reso_h*cropping_factor1[0] + 1) # 随机生成裁剪的起始y坐标
 
            contrast_factor1 = np.random.uniform(0.9,1.1, size = (2,)) # 随机生成对比度调整因子
            hue_factor1 = np.random.uniform(-0.05,0.05, size = (2,)) # 随机生成色调调整因子
            saturation_factor1 = np.random.uniform(0.9,1.1, size = (2,)) # 随机生成饱和度调整因子
            brightness_factor1 = np.random.uniform(0.9,1.1,size = (2,)) # 随机生成亮度调整因子
            gamma1 = np.random.uniform(0.85,1.15, size = (2,)) # 随机生成Gamma调整因子


            erase_size1 = np.random.randint(int(self.erase_size/2),self.erase_size, size = (2,)) # 随机生成擦除区域的大小
            erase_size2 = np.random.randint(int(self.erase_size/2),self.erase_size, size = (2,))
            random_color_dropped = np.random.randint(0,3,(2)) # 随机选择要丢弃的颜色通道

            # 循环读取视频帧
            while(cap.isOpened()): 
                count += 1
                ret, frame = cap.read() # 读取一帧
                if ((count not in frames_full)) and (ret == True): # 如果当前帧不在要提取的帧列表中，且读取成功，则跳过该帧
                    continue
                if ret == True: # 如果读取成功，且当前帧在要提取的帧列表中
                    if (count in frames_full): # 对该帧进行数据增强处理，并添加到完整片段列表中
                        full_clip.append(self.augmentation(frame, random_array[0], x_erase[0], y_erase[0], cropping_factor1[0],\
                                x0, y0, contrast_factor1[0], hue_factor1[0], saturation_factor1[0], brightness_factor1[0],\
                                gamma1[0],erase_size1[0],erase_size2[0], random_color_dropped[0]))
                        list_full.append(count) # 记录该帧的索引
                # 如果读取失败，退出循环
                else:
                    break 
            # 如果提取的帧数小于要求的帧数，但大于要求帧数的一半            
            if len(full_clip) < params.num_frames and len(full_clip)>(params.num_frames/2) :
                # 计算还需要的帧数
                remaining_num_frames = params.num_frames - len(full_clip)
                # 通过反转已提取的帧来补充缺失的帧数
                full_clip = full_clip + full_clip[::-1][1:remaining_num_frames+1]
            
            try:
                assert(len(full_clip)==params.num_frames)
                # 返回完整的视频片段
                return full_clip
            except:
                # 如果帧数不满足要求，打印错误信息
                print(f'Clip {vid_path} Failed')
                return None   

        except:
            # 如果出现异常，打印错误信息
            print(f'Clip {vid_path} Failed')
            return None
    # 数据增强
    def augmentation(self, image, random_array, x_erase, y_erase, cropping_factor1,\
        x0, y0, contrast_factor1, hue_factor1, saturation_factor1, brightness_factor1,\
        gamma1,erase_size1,erase_size2, random_color_dropped):
        
        image = self.PIL(image)
        image = trans.functional.resized_crop(image,y0,x0,int(params.ori_reso_h*cropping_factor1),int(params.ori_reso_h*cropping_factor1),(params.reso_h,params.reso_w))


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
            image = trans.functional.to_grayscale(image, num_output_channels = 3)
            if random_array[5] > 0.25:
                image = trans.functional.adjust_gamma(image, gamma = gamma1, gain=1) #gamma range [0.8, 1.2]
        if random_array[6] > 0.5:
            image = trans.functional.hflip(image)

        image = trans.functional.to_tensor(image)

        if random_array[6] < 0.5/2 :
            image = trans.functional.erase(image, x_erase, y_erase, erase_size1, erase_size2, v=0) 

        return image


class multi_baseline_dataloader_val_strong(Dataset):

    def __init__(self, shuffle = True, data_percentage = 1.0, mode = 0, skip = 1, \
                hflip=0, cropping_factor=1.0, split = 1):

        if split == 1:
            self.all_paths = open(os.path.join(cfg.path_folder, 'ucf101/ucfTrainTestlist/testlist01.txt'),'r').read().splitlines()
        elif split ==2: 
            self.all_paths = open(os.path.join(cfg.path_folder, 'ucf101/ucfTrainTestlist/testlist02.txt'),'r').read().splitlines()
        elif split ==3: 
            self.all_paths = open(os.path.join(cfg.path_folder, 'ucf101/ucfTrainTestlist/testlist03.txt'),'r').read().splitlines()
        else:
            print(f'Invalid split input: {split}')
        
        
        self.classes= json.load(open(cfg.class_mapping_ucf101))
        self.shuffle = shuffle

        if self.shuffle:
            random.shuffle(self.all_paths)
        
        self.data_percentage = data_percentage
        self.data_limit = int(len(self.all_paths)*self.data_percentage)
        self.data = self.all_paths[0: self.data_limit]
        self.PIL = trans.ToPILImage()
        self.TENSOR = trans.ToTensor()
        self.mode = mode
        self.skip = skip
        self.hflip = hflip
        self.cropping_factor = cropping_factor
                       
    
    def __len__(self):
        return len(self.data)
            
    def __getitem__(self,index):        
        clip, label, vid_path, frame_list = self.process_data(index)
        return clip, label, vid_path, frame_list

    def process_data(self, idx):
    
        # label_building
        vid_path = cfg.path_folder + 'ucf101/videos/' + self.data[idx].split(' ')[0]
        label = self.classes[vid_path.split('/')[-2]] # THIS MIGHT BE DIFFERNT AFTER STEVE MOVE THE PATHS  
        
        # clip_building
        clip, frame_list = self.build_clip(vid_path)

        return clip, label, vid_path, frame_list

    def build_clip(self, vid_path):
        try:
            cap = cv2.VideoCapture(vid_path) # 尝试打开视频文件
            cap.set(1, 0) # 设置视频读取的起始帧为第0帧
            frame_count = cap.get(7) # 获取视频的总帧数

            N = frame_count # 视频的总帧数
            n = params.num_frames # 要求提取的帧数

            # skip_max = int(N/n)
            # if (skip_max < 2) and (self.skip>1):
            #     # print(frame_count)
            #     # print('Skip_max =1 and already covered')
            #     return None


            skip_frames_full = params.fix_skip # 固定每帧之间的跳过帧数，从参数配置中获取
            # if skip_frames_full<2:
            #     if self.skip==1:
            #         skip_frames_full = 1
            #     else:
            #         # print('Skip_rate already covered')
            #         return None

            # 计算按照固定跳过帧数取帧后，剩余的帧数
            left_over = skip_frames_full*n 
            F = N - left_over

            # 根据模式计算起始帧
            start_frame_full = 0 + int(np.linspace(0,F-10,params.num_modes)[self.mode])

            # 如果起始帧小于0，将起始帧设为模式值
            if start_frame_full< 0:
                start_frame_full = self.mode

            # 生成要提取的帧的索引列表
            full_clip_frames = []
            full_clip_frames = start_frame_full + np.asarray(
                [int(int(skip_frames_full) * f) for f in range(params.num_frames)])
            
            # 当前帧的计数器
            count = -1

            full_clip = [] # 用于存储提取的帧
            list_full = [] # 用于存储实际提取的帧的索引

             # 循环读取视频帧
            while (cap.isOpened()):
                count += 1
                ret, frame = cap.read() # 读取一帧
                # print(frame.shape)
                if ((count not in full_clip_frames) and (ret == True)): # 如果当前帧不在要提取的帧列表中，且读取成功，则跳过该帧
                    continue
                # 如果读取成功，且当前帧在要提取的帧列表中
                if ret == True:
                    # 对该帧进行数据增强处理，并添加到完整片段列表中        
                    if (count in full_clip_frames):
                        full_clip.append(self.augmentation(frame))
                        list_full.append(count) # 记录该帧的索引
                else:
                    break
            # 如果提取的帧数小于要求的帧数，但大于要求帧数的一半            
            if len(full_clip) < params.num_frames and len(full_clip)>(params.num_frames/2):
                # 计算还需要的帧数
                remaining_num_frames = params.num_frames - len(full_clip)
                 # 通过反转已提取的帧来补充缺失的帧数
                full_clip = full_clip + full_clip[::-1][1:remaining_num_frames+1]
            # 断言提取的帧数等于要求的帧数
            assert (len(full_clip) == params.num_frames)
            # 返回完整的视频片段和实际提取的帧的索引列表
            return full_clip, list_full

        except:
            print(f'Clip {vid_path} Failed')
            return None, None

    def augmentation(self, image):
        image = self.PIL(image)
        

        if self.cropping_factor <= 1:
            image = trans.functional.center_crop(image,(int(params.ori_reso_h*self.cropping_factor),int(params.ori_reso_h*self.cropping_factor)))
        image = trans.functional.resize(image, (params.reso_h, params.reso_w))
        if self.hflip !=0:
            image = trans.functional.hflip(image)

        return trans.functional.to_tensor(image)


def collate_fn_val_ucf101(batch):

    f_clip, label, vid_path, frame_list = [], [], [], []
    # print(len(batch))
    for item in batch:
        if not (item[0] == None or item[1] == None or item[2] == None):
            f_clip.append(torch.stack(item[0],dim=0)) # I might need to convert this tensor to torch.float
            label.append(item[1])
            vid_path.append(item[2])
            frame_list.append(item[3])

        # else:
            # print('oh no2')
    # print(len(f_clip))
    f_clip = torch.stack(f_clip, dim=0)
    
    return f_clip, label, vid_path, frame_list 
            
def collate_fn_train_ucf101(batch):

    f_clip, label, vid_path = [], [], [], 
    # print(len(batch))
    for item in batch:
        if not (item[0] == None or item[1] == None or item[2] == None):
            f_clip.append(torch.stack(item[0],dim=0)) # I might need to convert this tensor to torch.float
            label.append(item[1])
            vid_path.append(item[2])
        # else:
            # print('oh no2')
    # print(len(f_clip))
    f_clip = torch.stack(f_clip, dim=0)
    
    return f_clip, label, vid_path 

if __name__ == '__main__':

    train_dataset = baseline_dataloader_train_strong(shuffle = False, data_percentage = 1.0)
    train_dataloader = DataLoader(train_dataset, batch_size=params.batch_size, \
        shuffle=True, num_workers=params.num_workers, collate_fn=collate_fn_train)

    print(f'Step involved: {len(train_dataset)/params.batch_size}')
    t=time.time()

    for i, (clip, label, vid_path) in enumerate(train_dataloader):
        if i%10 == 0:
            print()
            clip = clip.permute(0,1,3,4,2)
            print(f'Full_clip shape is {clip.shape}')
            print(f'Label is {label}')
            # pickle.dump(clip, open('f_clip.pkl','wb'))
            # pickle.dump(label, open('label.pkl','wb'))
            # exit()
    print(f'Time taken to load data is {time.time()-t}')


