import os
import cv2
import numpy as np
from tqdm import tqdm

def generate_defocus_images(input_dir, output_dir, sigma=3.0, kernel_size=None):
    """
    通过高斯模糊模拟 Defocus Lens 光学散焦效果

    Args:
        input_dir: 原始图像文件夹路径
        output_dir: 输出文件夹路径
        sigma: 高斯核标准差，值越大模糊越强，隐私保护越强但信息损失越大
               (论文参考: sigma=3.0 可产生显著隐私保护效果)
        kernel_size: 高斯核尺寸，若为 None 则自动根据 sigma 计算 (推荐保持 None)
    """
    os.makedirs(output_dir, exist_ok=True)

    # 收集所有图像文件
    image_files = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                image_files.append(os.path.join(root, f))

    # 自动确定核尺寸 (OpenCV 要求核尺寸为奇数)
    if kernel_size is None:
        kernel_size = int(2 * round(3 * sigma) + 1)  # 覆盖 99.7% 的能量

    for img_path in tqdm(image_files, desc="Generating Defocus Lens images"):
        # 读取图像
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Cannot read {img_path}, skipping...")
            continue

        # 高斯模糊模拟散焦
        img_defocused = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)

        # 保存图像，保持相对路径结构
        rel_path = os.path.relpath(img_path, input_dir)
        out_path = os.path.join(output_dir, rel_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, img_defocused)

    print(f"Done! Defocused images saved to {output_dir}")

def compute_psnr_batch(original_root, defocus_root):
    """计算原始图像与 Defocus 图像之间的 PSNR 统计值"""
    import numpy as np

    def compute_psnr(img1, img2, max_pixel=255.0):
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        return 10 * np.log10((max_pixel ** 2) / mse)

    psnr_values = []
    image_pairs = []

    # 收集图像对
    for root, _, files in os.walk(original_root):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                orig_path = os.path.join(root, f)
                rel_path = os.path.relpath(orig_path, original_root)
                defocus_path = os.path.join(defocus_root, rel_path)
                if os.path.exists(defocus_path):
                    image_pairs.append((orig_path, defocus_path))

    print(f"Found {len(image_pairs)} image pairs.")

    for orig_path, defocus_path in tqdm(image_pairs, desc="Computing PSNR"):
        img_orig = cv2.imread(orig_path)
        img_defocus = cv2.imread(defocus_path)
        if img_orig is None or img_defocus is None:
            continue
        if img_orig.shape != img_defocus.shape:
            img_defocus = cv2.resize(img_defocus, (img_orig.shape[1], img_orig.shape[0]))
        psnr = compute_psnr(img_orig, img_defocus)
        psnr_values.append(psnr)

    psnr_arr = np.array(psnr_values)
    print("\n====== PSNR Statistics (Defocus Lens vs Original) ======")
    print(f"Mean PSNR: {np.mean(psnr_arr):.2f} dB")
    print(f"Median PSNR: {np.median(psnr_arr):.2f} dB")
    print(f"Std: {np.std(psnr_arr):.2f} dB")
    print(f"Min/Max: {np.min(psnr_arr):.2f} / {np.max(psnr_arr):.2f} dB")
    return psnr_arr

# ========== 使用示例 ==========
if __name__ == "__main__":
    # 设置路径
    ORIGINAL_DIR = "/mnt/private/xwy/aed-mae/data/shanghaitech_AED/test/frames"      # 原始图像文件夹
    DEFOCUS_DIR = "/mnt/private/xwy/SPAct/anonymization_training/defocus_sh"        # Defocus 输出文件夹

    # Step 1: 生成 Defocus Lens 图像
    generate_defocus_images(ORIGINAL_DIR, DEFOCUS_DIR, sigma=3.0)

    # Step 2: 计算 PSNR 统计
    compute_psnr_batch(ORIGINAL_DIR, DEFOCUS_DIR)