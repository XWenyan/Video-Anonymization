import os
import cv2
import numpy as np
from tqdm import tqdm

def generate_low_resolution_images(input_dir, output_dir, scale_factor=0.5, method=cv2.INTER_CUBIC):
    """
    生成与原始图片尺寸相同的“模拟低分辨率”图片。
    """
    os.makedirs(output_dir, exist_ok=True)
    # 收集输入目录下的所有图片文件
    image_files = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, f))

    for img_path in tqdm(image_files, desc="Generating LR images"):
        # 1. 读取原始图片
        img_hr = cv2.imread(img_path)
        if img_hr is None:
            continue
        
        h, w = img_hr.shape[:2]

        # 2. 降采样到低分辨率
        lr_h, lr_w = int(h * scale_factor), int(w * scale_factor)
        img_lr = cv2.resize(img_hr, (lr_w, lr_h), interpolation=method)

        # 3. 上采样回原始尺寸
        img_sim_lr = cv2.resize(img_lr, (w, h), interpolation=method)

        # 4. 保存图片
        rel_path = os.path.relpath(img_path, input_dir)
        out_path = os.path.join(output_dir, rel_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, img_sim_lr)

    print(f"Done! Results saved to {output_dir}")

# --- 使用示例 ---
# 设置你的路径
original_folder = "/mnt/private/xwy/aed-mae/data/shanghaitech_AED/test/frames"
lowres_output_folder = "/mnt/private/xwy/SPAct/anonymization_training/lowr_sh"

# 调用函数生成模拟低分辨率图片
generate_low_resolution_images(original_folder, lowres_output_folder, scale_factor=0.25, method=cv2.INTER_CUBIC)