import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from tqdm import tqdm
from model import load_unet
import argparse

def preprocess_avenue_dataset(fa_model_path, input_root, output_root):
    """
    预处理Avenue数据集：将原始图片通过fa模型匿名化后保存
    
    参数:
        fa_model_path: fa模型路径
        input_root: Avenue数据集根目录 (包含testing/frames)
        output_root: 输出根目录
        target_size: 目标图像尺寸
    """
    target_size=(128, 128)
    # 加载fa模型
    fa_model = torch.load(fa_model_path)
    if isinstance(fa_model, dict):
        # 如果保存的是状态字典而非完整模型
        from model import load_unet
        model = load_unet()
        if 'fa_model_state_dict' in fa_model:
            state_dict = fa_model['fa_model_state_dict']
        else:
            state_dict = fa_model
        # 处理多GPU前缀
        if next(iter(state_dict)).startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        fa_model = model
    
    fa_model.eval()
    if torch.cuda.is_available():
        fa_model.cuda()
    
    # 创建输出目录结构
    input_folder = os.path.join(input_root, "test", "frames")
    output_folder = os.path.join(output_root, "testingsh", "frames")
    os.makedirs(output_folder, exist_ok=True)
    
    # 遍历所有视频文件夹
    video_folders = sorted([f for f in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, f))])
    
    print(f"开始处理Avenue数据集，共发现{len(video_folders)}个视频文件夹...")
    
    # 图像预处理转换
    preprocess = transforms.Compose([
        transforms.ToTensor(),  # 转换为张量并归一化到[0,1]
    ])
    
    for video in video_folders:
        input_video_path = os.path.join(input_folder, video)
        output_video_path = os.path.join(output_folder, video)
        os.makedirs(output_video_path, exist_ok=True)
        
        # 获取所有帧文件
        frame_files = sorted([f for f in os.listdir(input_video_path) if f.endswith('.png')])
        
        print(f"处理视频文件夹: {video}, 共{len(frame_files)}帧...")
        
        for frame_file in tqdm(frame_files, desc=f"处理 {video}"):
            # 读取原始图像
            img_path = os.path.join(input_video_path, frame_file)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转为RGB
            
            # 调整尺寸
            img_resized = cv2.resize(img, target_size)
            
            # 转换为张量 [0,1]
            img_tensor = preprocess(img_resized).unsqueeze(0)  # 添加批次维度
            
            # 移动到GPU（如果可用）
            if torch.cuda.is_available():
                img_tensor = img_tensor.cuda()
            
            # 通过fa模型
            with torch.no_grad():
                anonymized_img = fa_model(img_tensor)
            
            # 转换回numpy图像 [0,1] -> [0,255]
            anonymized_img = anonymized_img.squeeze(0).cpu().numpy()  # 移除批次维度
            anonymized_img = np.transpose(anonymized_img, (1, 2, 0))  # CHW -> HWC
            anonymized_img = (anonymized_img * 255).astype(np.uint8)
            
            # 保存匿名化后的图像
            output_path = os.path.join(output_video_path, frame_file)
            cv2.imwrite(output_path, cv2.cvtColor(anonymized_img, cv2.COLOR_RGB2BGR))
    
    print("预处理完成！匿名化后的图片已保存到:", output_folder)

if __name__ == "__main__":
    fa_model = '/mnt/private/xwy/SPAct/anonymization_training/saved_models/0616_shanghai/fa_model_72.pth'
    input_dir = '/mnt/private/xwy/aed-mae/data/shanghaitech_AED'
    output_dir = '/mnt/private/xwy/SPAct/anonymization_training/anonymized_shanghai'
    preprocess_avenue_dataset(fa_model, input_dir, output_dir)