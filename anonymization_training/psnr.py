import os
import cv2
import numpy as np
from tqdm import tqdm

def compute_psnr(img1, img2, max_pixel=255.0):
    """
    计算两张 RGB 图像的 PSNR
    img1, img2: numpy array, shape (H, W, 3), dtype uint8
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10((max_pixel ** 2) / mse)

def collect_image_pairs(orig_root, anon_root):
    """收集所有图像对，保证相对路径一致"""
    pairs = []
    for root, dirs, files in os.walk(orig_root):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
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
    original_root = "/mnt/private/xwy/aed-mae/data/avenue/test/frames"
    anonymized_root = "/mnt/private/xwy/SPAct/anonymization_training/ppopps_av2"
    output_csv = "psnr_results_sh_defocus.csv"  # 若不需要保存CSV，设为 None
    # ==========================================

    pairs = collect_image_pairs(original_root, anonymized_root)
    print(f"Found {len(pairs)} image pairs.")

    psnr_values = []
    results = []  # 用于保存详细结果

    for orig_path, anon_path in tqdm(pairs, desc="Computing PSNR"):
        # 读取图像 (BGR -> RGB)
        img_orig = cv2.imread(orig_path)
        img_anon = cv2.imread(anon_path)
        if img_orig is None or img_anon is None:
            print(f"Error reading: {orig_path} or {anon_path}")
            continue

        # 如果尺寸不一致，将匿名化图像缩放到原始尺寸
        if img_orig.shape != img_anon.shape:
            img_anon = cv2.resize(img_anon, (img_orig.shape[1], img_orig.shape[0]))

        psnr = compute_psnr(img_orig, img_anon)
        psnr_values.append(psnr)
        results.append((os.path.relpath(orig_path, original_root), psnr))

    # 统计
    psnr_arr = np.array(psnr_values)
    mean_psnr = np.mean(psnr_arr)
    median_psnr = np.median(psnr_arr)
    std_psnr = np.std(psnr_arr)
    min_psnr = np.min(psnr_arr)
    max_psnr = np.max(psnr_arr)

    print("\n====== PSNR Statistics ======")
    print(f"Mean PSNR: {mean_psnr:.2f} dB")
    print(f"Median PSNR: {median_psnr:.2f} dB")
    print(f"Standard deviation: {std_psnr:.2f} dB")
    print(f"Min PSNR: {min_psnr:.2f} dB")
    print(f"Max PSNR: {max_psnr:.2f} dB")
    print(f"Total pairs: {len(psnr_values)}")

    # 保存详细结果到 CSV
    if output_csv:
        import csv
        with open(output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["relative_path", "PSNR_dB"])
            writer.writerows(results)
        print(f"Detailed results saved to {output_csv}")

if __name__ == "__main__":
    main()