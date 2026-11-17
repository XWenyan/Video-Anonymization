import os
import cv2
import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

def compute_ssim(img1, img2, use_gray=True):
    """
    计算两张图像的 SSIM 值
    img1, img2: numpy array, H x W x C (RGB) 或 H x W (灰度)
    use_gray: 是否转为灰度图计算（若为彩色图且use_gray=True，则转为灰度）
    若 use_gray=False，则对每个通道计算后取平均
    """
    if use_gray:
        if len(img1.shape) == 3:
            img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        else:
            img1_gray = img1
            img2_gray = img2
        ssim_val = ssim(img1_gray, img2_gray, data_range=255)
    else:
        # 多通道 SSIM (RGB)
        ssim_val = ssim(img1, img2, multichannel=True, data_range=255, channel_axis=2)
    return ssim_val

def collect_image_pairs(orig_root, anon_root, ext=('.png', '.jpg', '.jpeg')):
    """收集所有图像对，保证相对路径一致"""
    pairs = []
    for root, _, files in os.walk(orig_root):
        for f in files:
            if f.lower().endswith(ext):
                orig_path = os.path.join(root, f)
                rel_path = os.path.relpath(orig_path, orig_root)
                anon_path = os.path.join(anon_root, rel_path)
                if os.path.exists(anon_path):
                    pairs.append((orig_path, anon_path))
                else:
                    print(f"Warning: No matching anonymized image for {orig_path}")
    return pairs

def main():
    # ================= 设置路径 =================
    original_root = "/mnt/private/xwy/aed-mae/data/shanghaitech_AED/test/frames"
    anonymized_root = "/mnt/private/xwy/SPAct/anonymization_training/ppopps_sh2"
    output_csv = "ssim_results_av0415.csv"   # 若不需要保存CSV，设为 None
    use_gray = True                   # 是否使用灰度图像计算 SSIM（推荐 True）
    # ==========================================

    pairs = collect_image_pairs(original_root, anonymized_root)
    print(f"Found {len(pairs)} image pairs.")

    ssim_values = []
    results = []  # 保存详细结果（相对路径, SSIM）

    for orig_path, anon_path in tqdm(pairs, desc="Computing SSIM"):
        img_orig = cv2.imread(orig_path)
        img_anon = cv2.imread(anon_path)
        if img_orig is None or img_anon is None:
            print(f"Error reading: {orig_path} or {anon_path}")
            continue

        # 如果尺寸不一致，将匿名化图像缩放到原始尺寸
        if img_orig.shape != img_anon.shape:
            img_anon = cv2.resize(img_anon, (img_orig.shape[1], img_orig.shape[0]))

        ssim_val = compute_ssim(img_orig, img_anon, use_gray=use_gray)
        ssim_values.append(ssim_val)
        results.append((os.path.relpath(orig_path, original_root), ssim_val))

    # 统计
    ssim_arr = np.array(ssim_values)
    mean_ssim = np.mean(ssim_arr)
    median_ssim = np.median(ssim_arr)
    std_ssim = np.std(ssim_arr)
    min_ssim = np.min(ssim_arr)
    max_ssim = np.max(ssim_arr)

    print("\n====== SSIM Statistics ======")
    print(f"Mean SSIM: {mean_ssim:.4f}")
    print(f"Median SSIM: {median_ssim:.4f}")
    print(f"Standard deviation: {std_ssim:.4f}")
    print(f"Min SSIM: {min_ssim:.4f}")
    print(f"Max SSIM: {max_ssim:.4f}")
    print(f"Total pairs: {len(ssim_values)}")

    # 保存详细结果到 CSV
    if output_csv:
        import csv
        with open(output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["relative_path", "SSIM"])
            writer.writerows(results)
        print(f"Detailed results saved to {output_csv}")

if __name__ == "__main__":
    main()