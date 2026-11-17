import chardet

with open('sht_test_frames_list.txt', 'rb') as f:
    result = chardet.detect(f.read())  # 读取一定量的数据进行编码检测

print(result['encoding'])  # 打印检测到的编码
