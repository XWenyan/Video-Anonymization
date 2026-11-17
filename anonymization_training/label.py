import os
import json
import pickle
import concurrent.futures
from tqdm import tqdm

# 定义JSON文件所在的目录
# json_dir = "/opt/data/common/VPA/annotations/train2017"
json_dir = "/mnt/private/xwy/Balancing_privacy_and_pose/dataset/vispr/test2017_anno"

# 初始化一个字典来存储标签信息
labels_dict = {}

# 加载单个JSON文件并提取标签信息的函数
def process_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
        image_id = data['id']
        labels = data['labels']
        return image_id, labels

# 获取所有JSON文件的路径
json_files = [os.path.join(json_dir, f) for f in os.listdir(json_dir) if f.endswith('.json')]

# 使用多线程处理JSON文件
with concurrent.futures.ThreadPoolExecutor() as executor:
    # 使用tqdm显示进度条
    results = list(tqdm(executor.map(process_json_file, json_files), total=len(json_files), desc="Processing JSON files"))

# 将结果存储到标签字典中
for image_id, labels in results:
    labels_dict[image_id] = labels

# 将标签字典保存到Pickle文件中
with open(os.path.join(os.getcwd(), 'test_labels.pkl'), 'wb') as f:
    pickle.dump(labels_dict, f)

print("Labels have been saved to test_labels.pkl")