import torch
from collections import OrderedDict
import argparse

def extract_models(input_path, output_dir):
    """
    从保存的检查点中提取三个模型的状态字典
    
    参数:
        input_path: 包含三个模型的.pth文件路径
        output_dir: 输出目录，用于保存提取的模型
    """
    # 加载整个检查点
    checkpoint = torch.load(input_path)
    
    # 确保输出目录存在
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 提取并保存fa_model (UNet)
    if 'fa_model_state_dict' in checkpoint:
        fa_state_dict = checkpoint['fa_model_state_dict']
        # 去除多GPU前缀 (如果存在)
        new_fa_state_dict = OrderedDict()
        for k, v in fa_state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_fa_state_dict[name] = v
        
        # 保存fa_model
        fa_output_path = os.path.join(output_dir, 'fa_model.pth')
        torch.save({
            'fa_model_state_dict': new_fa_state_dict,
            'epoch': checkpoint.get('epoch', 0)
        }, fa_output_path)
        print(f"Saved fa_model to {fa_output_path}")
    
    # 提取并保存fb_model (隐私分类器)
    if 'fb_model_state_dict' in checkpoint:
        fb_state_dict = checkpoint['fb_model_state_dict']
        # 去除多GPU前缀
        new_fb_state_dict = OrderedDict()
        for k, v in fb_state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_fb_state_dict[name] = v
        
        # 保存fb_model
        fb_output_path = os.path.join(output_dir, 'fb_model.pth')
        torch.save({
            'fb_model_state_dict': new_fb_state_dict,
            'epoch': checkpoint.get('epoch', 0)
        }, fb_output_path)
        print(f"Saved fb_model to {fb_output_path}")
    
    # 提取并保存ft_model (异常检测模型)
    if 'ft_model_state_dict' in checkpoint:
        ft_state_dict = checkpoint['ft_model_state_dict']
        # 去除多GPU前缀
        new_ft_state_dict = OrderedDict()
        for k, v in ft_state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_ft_state_dict[name] = v
        
        # 保存ft_model
        ft_output_path = os.path.join(output_dir, 'ft_model.pth')
        torch.save({
            'ft_model_state_dict': new_ft_state_dict,
            'epoch': checkpoint.get('epoch', 0),
            'm_items': checkpoint.get('m_items', None)  # 保存关联的内存项
        }, ft_output_path)
        print(f"Saved ft_model to {ft_output_path}")
    
    # 单独保存内存项 (如果需要)
    if 'm_items' in checkpoint:
        m_items = checkpoint['m_items']
        m_output_path = os.path.join(output_dir, 'm_items.pt')
        torch.save(m_items, m_output_path)
        print(f"Saved memory items to {m_output_path}")

if __name__ == '__main__':
    input_path = '/opt/data/private/xwy/SPAct/anonymization_training/saved_models/train0626/model_6.pth'
    output_path = '/opt/data/private/xwy/SPAct/anonymization_training/saved_models/train0626'
    extract_models(input_path, output_path)