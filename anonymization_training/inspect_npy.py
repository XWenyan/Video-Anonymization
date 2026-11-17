import numpy as np
import sys
import os

def inspect_npy(filepath):
    """加载 .npy 文件并打印关键信息"""
    if not os.path.exists(filepath):
        print(f"❌ 文件不存在: {filepath}")
        return

    data = np.load(filepath)
    print(f"\n📁 文件: {filepath}")
    print(f"   - Shape : {data.shape}")
    print(f"   - Dtype : {data.dtype}")
    print(f"   - 维度数: {data.ndim}")
    
    if data.ndim == 1:
        print(f"   - 前10个元素: {data[:10]}")
    elif data.ndim == 2:
        if data.shape[0] > 0:
            print(f"   - 第0行 shape: {data[0].shape}")
            print(f"   - 第0行前10个: {data[0][:10]}")
        else:
            print("   - 第0行不存在（行数为0）")
    else:
        print(f"   - 高维数组，仅显示整体形状: {data.shape}")
    print("   -" + "-"*50)

if __name__ == "__main__":
    # 用法：命令行传入文件路径，或直接在下方修改路径
    if len(sys.argv) > 1:
        inspect_npy(sys.argv[1])
    else:
        # 此处请替换为你实际要检查的 .npy 文件路径
        # 示例（请根据你的实际情况修改）：
        files_to_check = [
            "/mnt/private/xwy/SPAct/datasets/frame_labels_avenue.npy",
            "/mnt/private/xwy/SPAct/datasets/frame_labels_shanghai.npy",
            # 你也可以添加 avenue 的标签文件，对比看
        ]
        for f in files_to_check:
            inspect_npy(f)