import os
import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

# 假设你的 UNet 定义在 mnad_model.final_future_prediction_with_memory_spatial_sumonly_weight_ranking_top1 中
# 如果无法直接导入，请确保模型定义文件在 Python 路径中，或在此处手动定义模型结构。
# 这里提供一种通用的加载方式：直接 torch.load 整个模型（因为训练代码中保存的是完整模型）
from models.unet_model import UNet

def load_fa_model(model_path, device):
    """
    加载训练好的匿名化模型 (fa_model)
    训练代码中保存格式：torch.save(fa_model, os.path.join(save_dir, 'fa_model_{}.pth'.format(epoch)))
    因此可以直接 load 整个模型对象。
    """
    print(f"Loading fa_model from {model_path}")
    model = torch.load(model_path, map_location=device)
    model = model.to(device)
    model.eval()
    return model

def anonymize_image(model, image_tensor, device):
    """
    对单张图像张量进行匿名化
    Args:
        model: 加载好的 fa_model
        image_tensor: shape [1, 3, H, W], 值域 [0, 1], 设备 CPU/GPU
        device: torch.device
    Returns:
        anonymized_tensor: shape [1, 3, H, W], 值域 [0, 1]
    """
    with torch.no_grad():
        output = model(image_tensor.to(device))
    return output.cpu()

def save_tensor_as_image(tensor, save_path):
    """
    将 [0,1] 范围的图像张量保存为 PNG 文件
    """
    # 去掉 batch 维度，转为 HWC numpy，值域 [0,255]
    img_np = tensor.squeeze(0).permute(1, 2, 0).numpy()
    img_np = (img_np * 255).astype(np.uint8)
    Image.fromarray(img_np).save(save_path)

def anonymize_folder(model, input_dir, output_dir, device, image_extensions=('.png', '.jpg', '.jpeg', '.bmp')):
    """
    批量匿名化文件夹中的所有图像
    """
    os.makedirs(output_dir, exist_ok=True)
    # 收集所有图像文件
    image_files = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith(image_extensions):
                image_files.append(os.path.join(root, f))
    
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转为 [0,1] 的 Tensor，形状 HWC -> CHW
    ])
    
    for img_path in tqdm(image_files, desc="Anonymizing"):
        # 构建输出路径（保持相对目录结构）
        rel_path = os.path.relpath(img_path, input_dir)
        out_path = os.path.join(output_dir, rel_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        
        # 加载并预处理
        pil_img = Image.open(img_path).convert('RGB')
        img_tensor = transform(pil_img).unsqueeze(0)  # [1,3,H,W]
        
        # 匿名化
        anon_tensor = anonymize_image(model, img_tensor, device)
        
        # 保存
        save_tensor_as_image(anon_tensor, out_path)

def main():
    # ========== 在这里设置你的默认路径 ==========
    model_path = "/mnt/private/xwy/SPAct/anonymization_training/saved_models/train0406/fa_model_48.pth"      # 替换为实际模型路径
    input_dir  = "/mnt/private/xwy/aed-mae/data/avenue/test/frames"       # 测试图像文件夹
    output_dir = "/mnt/private/xwy/SPAct/anonymization_training/fa_result/avenue0406" # 输出文件夹
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ============================================

    print(f"Using device: {device}")
    model = load_fa_model(model_path, device)
    anonymize_folder(model, input_dir, output_dir, device)
    print(f"Done. Results saved to {output_dir}")
if __name__ == "__main__":
    main()