#srun -c12 --mem-per-cpu=7G --pty -p gpu --gres=gpu:ampere:1 --pty bash
#sbatch -c12 --mem-per-cpu=7G -p gpu --wrap="python train_ssl_minimax2.py --run_id="" --kin_pretrained" --gres=gpu:ampere:1 --job-name="ours" --output=".out"


import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import *
import torchvision.transforms as transforms


import time, random, os, cv2, math, argparse, itertools, warnings, sys, traceback
import numpy as np 

import parameters_SSL as params
import config as cfg
# from keras.utils import to_categorical


from sklearn.metrics import precision_recall_fscore_support, average_precision_score
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from dl_vispr_ssl import vispr_ssl, collate_fn_train
from dl_vispr import vispr_dataset_generator
from dl_vispr import collate_fn_train as collate_fn_train_regular
# 加载数据集
from vad_utlis import vadDataLoader
# 加载模型
from mnad_model.final_future_prediction_with_memory_spatial_sumonly_weight_ranking_top1 import *
from model import build_r3d_classifier, load_r3d_classifier, ssl_pretrained_load, load_unet, build_privacy_classifier, load_privacy_classifier, ssl_privacy_load
# 加载损失函数
from nt_xent_original import NTXentLoss
from torchvision.utils import save_image
#
from utils import *


from collections import OrderedDict
import glob

warnings.filterwarnings("ignore", category=DeprecationWarning) 

torch.backends.cudnn.benchmark = True


from torch.autograd import Function

#=========梯度反转模块==========
class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


def grad_reverse(x, lambda_=1.0):
    return GradReverse.apply(x, lambda_)




# ========== 新增 VAD 验证函数 ==========
def val_epoch_vad(run_id, epoch, fa_model, ft_model, m_items, test_folder, labels_path, args_vad, use_cuda):
    """
    评估 VAD 性能（AUC）
    :param run_id: 运行ID（用于日志）
    :param epoch: 当前epoch
    :param fa_model: 匿名化模型（UNet）
    :param ft_model: 异常检测模型（convAE）
    :param m_items: 当前内存项张量（形状 [msize, mdim]）
    :param test_folder: Avenue测试集路径
    :param labels_path: 帧级标签文件路径
    :param args_vad: 包含VAD评估参数的简单对象
    :param use_cuda: 是否使用GPU
    :return: AUC值
    """
    fa_model.eval()
    ft_model.eval()

    # 加载测试数据集
    test_dataset = vadDataLoader(test_folder, transforms.Compose([transforms.ToTensor()]),
                                 resize_height=args_vad.h, resize_width=args_vad.w,
                                 time_step=args_vad.t_length - 1)
    test_batch = DataLoader(test_dataset, batch_size=args_vad.test_batch_size,
                            shuffle=False, num_workers=args_vad.num_workers_test, drop_last=False)

    # 加载帧级标签
    labels = np.load(labels_path)

    # 获取所有视频目录
    videos = OrderedDict()
    videos_list = sorted(glob.glob(os.path.join(test_folder, '*')))
    for video in videos_list:
        video_name = video.split('/')[-1]
        videos[video_name] = {}
        videos[video_name]['path'] = video
        videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.png'))
        videos[video_name]['frame'].sort()
        videos[video_name]['length'] = len(videos[video_name]['frame'])

    # 分配标签
    labels_list = []
    label_length = 0
    psnr_list = {}
    feature_distance_list = {}
    for video in sorted(videos_list):
        video_name = video.split('/')[-1]
        if args_vad.method == 'pred':
            labels_list = np.append(labels_list, labels[0][4 + label_length:videos[video_name]['length'] + label_length])
        else:
            labels_list = np.append(labels_list, labels[0][label_length:videos[video_name]['length'] + label_length])
        label_length += videos[video_name]['length']
        psnr_list[video_name] = []
        feature_distance_list[video_name] = []

    # 复制内存项（测试时动态更新使用副本）
    m_items_test = m_items.clone().detach()

    label_length = 0
    video_num = 0
    label_length += videos[videos_list[video_num].split('/')[-1]]['length']

    loss_func_mse = nn.MSELoss(reduction='none')

    for k, (imgs) in enumerate(test_batch):
        # 更新当前视频索引
        if args_vad.method == 'pred':
            if k == label_length - 4 * (video_num + 1):
                video_num += 1
                label_length += videos[videos_list[video_num].split('/')[-1]]['length']
        else:
            if k == label_length:
                video_num += 1
                label_length += videos[videos_list[video_num].split('/')[-1]]['length']

        if use_cuda:
            imgs = imgs.cuda()

        # 匿名化：fa_model 处理
        B, C_times, H, W = imgs.shape
        T = args_vad.t_length
        C = 3
        imgs_reshaped = imgs.view(B, T, C, H, W)
        imgs_flat = imgs_reshaped.view(B * T, C, H, W)
        anon_flat = fa_model(imgs_flat)  # 输出 [B*T, C, H, W]
        anon_reshaped = anon_flat.view(B, T, C, H, W)

        # 准备 ft_model 输入：前4帧拼接为12通道，第5帧作为目标
        input_ft = anon_reshaped[:, :4, :, :, :].contiguous().view(B, 4 * C, H, W)
        target_ft = anon_reshaped[:, 4, :, :, :]

        # ft_model 前向传播（update=False，但返回更新后的 m_items_test 副本）
        if args_vad.method == 'pred':
            outputs, feas, updated_feas, m_items_test, softmax_score_query, softmax_score_memory, _, _, _, compactness_loss = ft_model.forward(input_ft, m_items_test, False)
        else:
            outputs, feas, updated_feas, m_items_test, softmax_score_query, softmax_score_memory, compactness_loss = ft_model.forward(input_ft, m_items_test, False)

        # 计算 MSE
        mse_imgs = torch.mean(loss_func_mse(outputs, target_ft)).item()
        mse_feas = compactness_loss.mean().item() if compactness_loss.dim() > 0 else compactness_loss.item()

        # 点级分数
        point_sc = point_score(outputs, target_ft)

        # 动态更新内存项（如果分数低于阈值）
        if point_sc < args_vad.th:
            query = F.normalize(feas, dim=1)
            query = query.permute(0, 2, 3, 1)
            # 注意：此处直接调用 model.memory.update，需要传入当前 m_items_test
            if hasattr(ft_model, 'memory'):
                m_items_test = ft_model.memory.update(query, m_items_test, False)
            else:
                m_items_test = ft_model.module.memory.update(query, m_items_test, False)

        video_name = videos_list[video_num].split('/')[-1]
        psnr_list[video_name].append(psnr(mse_imgs))
        feature_distance_list[video_name].append(mse_feas)

    # 计算异常分数和 AUC
    anomaly_score_total_list = []
    for video in sorted(videos_list):
        video_name = video.split('/')[-1]
        anomaly_score_total_list += score_sum(
            anomaly_score_list(psnr_list[video_name]),
            anomaly_score_list_inv(feature_distance_list[video_name]),
            args_vad.alpha
        )
    anomaly_score_total_list = np.asarray(anomaly_score_total_list)
    auc = AUC(anomaly_score_total_list, np.expand_dims(1 - labels_list, 0))
    print(f'VAD Evaluation at epoch {epoch}: AUC = {auc*100:.2f}%')
    return auc


def process_avenue_data(inputs_avenue, fa_model):
    """处理并恢复 Avenue 数据格式"""
    B, C, H, W = inputs_avenue.shape
    
    # 重塑为 [B, T, C_frame, H, W] = [4, 5, 3, 128, 128]
    reshaped = inputs_avenue.view(B, 5, 3, H, W)
    
    # 转换为 [B, T, H, W, C] 并合并批次
    temporal_input = reshaped.permute(0, 1, 3, 4, 2)  # [4,5,128,128,3]
    combined = temporal_input.reshape(-1, H, W, 3)    # [20,128,128,3]
    fa_input = combined.permute(0, 3, 1, 2)           # [20,3,128,128]
    
    # 通过模型
    fa_output = fa_model(fa_input)  # [20,3,128,128]
    
    # 恢复原始格式
    output_restored = fa_output.permute(0, 2, 3, 1)    # [20,128,128,3]
    output_restored = output_restored.reshape(B, 5, H, W, 3)
    output_restored = output_restored.permute(0, 1, 4, 2, 3)  # [4,5,3,128,128]
    final_output = output_restored.reshape(B, 15, H, W)  # [4,15,128,128]
    
    return final_output


def train_epoch(run_id, epoch, data_loader_vispr, dataloader_avenue, ft_model, fa_model, fb_model, m_items, criterion_ft, criterion_fb, optimizer_fa, optimizer_fb, optimizer_ft, writer, use_cuda,learning_rate_fa, learning_rate_fb, learning_rate_ft,ft_w):
      
    # 学习率设置与日志同步
    for param_group in optimizer_fa.param_groups:
        param_group['lr']=learning_rate_fa
    for param_group_fb in optimizer_fb.param_groups:
        param_group_fb['lr']=learning_rate_fb
    for param_group_ft in optimizer_ft.param_groups:
        param_group_ft['lr']=learning_rate_ft
    writer.add_scalar('Learning Rate', learning_rate_fa, epoch)  
    writer.add_scalar('Learning Rate', learning_rate_fb, epoch)  
    writer.add_scalar('Learning Rate', learning_rate_ft, epoch)  

    print("Learning rate of fa is: {}".format(param_group['lr']))
    print("Learning rate of fb is: {}".format(param_group_fb['lr']))
    print("Learning rate of ft is: {}".format(param_group_ft['lr']))

    # 损失初始化与数据遍历
    losses_fa, losses_fb, losses_ft = [], [], []

    step = 1 # 交替优化的步骤标记（1更新fa,2更新fb和ft）
    for i, (data1, data2) in enumerate(zip(data_loader_vispr, dataloader_avenue)):
        if use_cuda:
            # Avenue数据保持[-1,1]范围
            inputs_avenue = data2          
            inputs_avenue = Variable(inputs_avenue).cuda() 
            # VISPR数据保持[0,1]范围
            inputs_vispr = [data1[j].cuda() for j in range(2)] # inputs_vispr是列表，包含img1和img2
            
        optimizer_fa.zero_grad()
        optimizer_fb.zero_grad()
        optimizer_ft.zero_grad()

        # 更新fa模型
        if step == 1:
            fa_model.train() # fa为训练模式
            ft_model.eval() # ft和fb为评估模式
            fb_model.eval()

            # 计算fb对fa输出的两个视图的结果，计算对比损失            
            output1 = [fb_model(fa_model(inputs_vispr[ii])) for ii in range(2)] # output1用于fb
            con_loss_criterion = NTXentLoss(device='cuda', batch_size=output1[0].shape[0], temperature=0.1, use_cosine_similarity=False)
            loss_fb = con_loss_criterion(output1[0], output1[1]) # 计算对比损失

            # 计算异常检测损失
            # 转换1: Avenue数据输入fa前 [-1,1] => [0,1]
            inputs_avenue_for_fa = (inputs_avenue + 1) / 2.0
            output2 = process_avenue_data(inputs_avenue_for_fa, fa_model)
            # 转换2: fa输出 => ft输入 [0,1] => [-1,1]
            output2_for_ft = output2 * 2 - 1

            outputs, feas, updated_feas, m_items, _, _, separateness_loss, compactness_loss = \
                ft_model.forward(output2_for_ft[:, 0:12], m_items, True)
            loss_pixel = torch.mean(criterion_ft(outputs, output2[:,12:]))
           
            # 确保损失为标量
            compactness_loss = compactness_loss.mean() if compactness_loss.dim() > 0 else compactness_loss
            separateness_loss = separateness_loss.mean() if separateness_loss.dim() > 0 else separateness_loss
           
            # 计算总体损失
            loss_ft = loss_pixel + params.loss_compact * compactness_loss + params.loss_separate * separateness_loss 
          
            #===============VAD-aware consistency======================
            # 1. 提取原始特征（未匿名）
            with torch.no_grad():
                out_orig = ft_model.forward(inputs_avenue_for_fa[:, 0:12], m_items, True)
            feas_orig = out_orig[1]
            feas_anon = feas 
            feat_diff = (feas_orig - feas_anon) ** 2

            # ---------- (A) Prediction mask ----------
            pred_error = torch.mean((outputs - output2[:,12:])**2, dim=1, keepdim=True)
            pred_error = (pred_error - pred_error.min()) / (pred_error.max() - pred_error.min() + 1e-6)
            normal_mask = torch.exp(-pred_error)

            # ---------- (B) Motion mask ----------
            B, _, H, W = output2_for_ft.shape
            # 3. 特征差（表示信息改变程度）
            feat_diff = (feas_orig - feas_anon) ** 2   # [B,C,H,W]
            frames = output2_for_ft[:, 0:12].view(B, 4, 3, H, W)
            
            motion = torch.mean((frames[:,1:] - frames[:,:-1])**2, dim=2)
            motion = torch.mean(motion, dim=1, keepdim=True)
            motion = (motion - motion.min()) / (motion.max() - motion.min() + 1e-6)
            # ---------- (C) Temporal mask ----------
            if feas_anon.shape[0] > 1:
                temp_diff = (feas_anon[1:] - feas_anon[:-1])**2
                temp_diff = torch.mean(temp_diff, dim=1, keepdim=True)

                temp_diff = (temp_diff - temp_diff.min()) / (temp_diff.max() - temp_diff.min() + 1e-6)

                temp_mask = torch.exp(-temp_diff)

                # pad 回 batch size
                temp_mask = torch.cat([temp_mask, temp_mask[-1:].clone()], dim=0)
            else:
                temp_mask = torch.ones_like(normal_mask)

            # ---------- (D) Final mask ----------
            normal_mask = F.interpolate(normal_mask, size=feat_diff.shape[-2:], mode='bilinear', align_corners=False)
            motion = F.interpolate(motion, size=feat_diff.shape[-2:], mode='bilinear', align_corners=False)
            
            mask = normal_mask * motion * temp_mask
            
            mask = torch.clamp(mask, 0.05, 1.0) 
            mask = mask.detach()
            # ---------- (E) Consistency loss ---------- 
            loss_cons = (feat_diff * mask).mean()

            # ===================== FA LOSS =====================
            loss_fa = -params.fb_loss_weight * loss_fb \
            + ft_w * loss_ft \
            + 2.0 * loss_cons

            losses_fa.append(loss_fa.item()) # 记录损失值
            
            # 反向传播并更新fa模型（仅fa的梯度）
            loss_fa.backward()
            optimizer_fa.step()
            if i % 100 ==0:
                for name, p in fa_model.named_parameters():
                    if p.grad is not None:
                        print("fa grad:", p.grad.abs().mean().item())
                        break


            # 切换到step2，接下来更新fb和ft
            step = 2
            if i % 100 == 0: # 每100批次打印中间结果
                print("feat_diff:", feat_diff.mean().item())
                print(mask.mean().item(), mask.std().item())
                print("loss_cons:", loss_cons.item())
                print(f'Training Epoch {epoch}, Batch {i}, loss_fa: {np.mean(losses_fa) :.5f}', flush = True)

            continue # 跳过step2

        if step == 2:
            fa_model.eval()
            ft_model.train()
            fb_model.train()
            # 同Step 1，计算fb对fa输出的两个视图的结果（对比损失输入）
            output1 = [fb_model(fa_model(inputs_vispr[ii])) for ii in range(2)] # output1用于fb
            con_loss_criterion = NTXentLoss(device = 'cuda', batch_size= output1[0].shape[0], temperature= 0.1, use_cosine_similarity= False) # 初始化对比损失
            loss_fb = con_loss_criterion(output1[0], output1[1]) # 计算对比损失
            
            # 同Step 1，计算异常检测损失
            # 转换1: Avenue数据输入fa前 [-1,1] => [0,1]
            inputs_avenue_for_fa = (inputs_avenue + 1) / 2.0
            output2 = process_avenue_data(inputs_avenue_for_fa, fa_model)
            # 转换2: fa输出 => ft输入 [0,1] => [-1,1]
            output2_for_ft = output2 * 2 - 1
            outputs, _, _, m_items, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss = ft_model.forward(output2_for_ft[:,0:12], m_items, True)
            loss_pixel = torch.mean(criterion_ft(outputs, output2[:,12:]))
           
            # 确保损失为标量
            compactness_loss = compactness_loss.mean() if compactness_loss.dim() > 0 else compactness_loss
            separateness_loss = separateness_loss.mean() if separateness_loss.dim() > 0 else separateness_loss

            # 计算总体损失
            loss_ft = loss_pixel + params.loss_compact * compactness_loss + params.loss_separate * separateness_loss 
            
            losses_ft.append(loss_ft.item())
            losses_fb.append(loss_fb.item())

            # 反向传播fb
            loss_fb.backward()
            optimizer_fb.step()
            # 反向传播ft
            loss_ft.backward(retain_graph=True)
            optimizer_ft.step()

            #切换回1，下次更新fa
            step = 1
            if (i-1) % 100 == 0: 
                print(f'Training Epoch {epoch}, Batch {i}, loss_fb: {np.mean(losses_fb) :.5f}, loss_ft: {np.mean(losses_ft) :.5f}', flush = True)
                
            continue
          
    print('Training Epoch: %d, loss_fa: %.4f, loss_fb: %.4f, loss_ft: %.4f' % (epoch,  np.mean(losses_fa), np.mean(losses_fb), np.mean(losses_ft)))
    writer.add_scalar('Training loss_fa', np.mean(losses_fa), epoch)
    writer.add_scalar('Training loss_fb', np.mean(losses_fb), epoch)
    writer.add_scalar('Training loss_ft', np.mean(losses_ft), epoch)

    del loss_fb, loss_ft, loss_fa,  inputs_vispr, inputs_avenue,  output1, output2

    return fa_model, fb_model, ft_model, m_items, ft_w

# 可视化特征提取器 fa_model 的输入与输出，用于验证阶段的结果分析。
def val_visualization_fa_vispr(save_dir, epoch, validation_dataloader, fa_model): # 保存目录 训练轮数 数据加载器 待可视化的模型
    for i, (inputs, label, vid_path) in enumerate(validation_dataloader):
    # inputs = inputs.permute(0,4,1,2,3)
        # 跳过无效输入（如形状为1维的样本，可能是数据加载异常）
        if len(inputs.shape) == 1:
            continue

        # 将输入数据移动到GPU（如果模型在GPU上，但此处未显式判断use_cuda，假设fa_model已在GPU）
        inputs = inputs.cuda()
        # 生成保存的文件名，格式为：save_dir/combined_e{epoch}.png
        image_full_name = save_dir + 'combined_e'+ str(epoch) + '.png'
        # print(inputs.shape)
        with torch.no_grad(): # 禁用梯度计算，节省内存并加速       
            outputs = fa_model(inputs) # fa_model前向传播，输出特征图或重建后的图像
            vis_image = torch.cat([inputs, outputs], dim=0) # 在维度0（批次维度）上拼接输入和输出，便于横向对比
            save_image(vis_image, image_full_name, padding=5, nrow=int(inputs.shape[0]))  # 保存图像：使用torchvision.utils.save_image，支持批量图像网格排列

            # if i ==0:
            break

# 在 VISPR 验证集上评估隐私分类任务的性能，计算二分类指标（F1、AP 等）。
def val_epoch_vispr(run_id, epoch, validation_dataloader, fa_model, fb_model, criterion, writer, use_cuda):
    
    fa_model.eval() # 特征提取器设为评估模式（关闭dropout/batchnorm统计更新）
    fb_model.eval() # 隐私分类器设为评估模式
    losses = [] # 记录每个批次的损失值
    predictions, ground_truth = [], [] # 收集所有样本的预测值（logits）和真实标签
    vid_paths = []  # 收集所有样本的视频路径
    label_dict, pred_dict = {}, {}

    for i, (inputs, label, vid_path) in enumerate(validation_dataloader):
        # inputs = inputs.permute(0,4,1,2,3)
        # 跳过无效输入，数据移动至GPU
        if len(inputs.shape) != 1:
            # inputs = inputs.permute(0, 2, 1, 3, 4)
            if use_cuda:
                inputs = inputs.cuda() # 输入帧序列（B,T,C,H,W）
                label = torch.from_numpy(np.asarray(label)).float().cuda() # 标签转为float型tensor  
            
            with torch.no_grad():# 禁用梯度计算，节省内存并加速  
                output1 = fa_model(inputs)  # 特征提取：fa_model处理输入帧序列，输出特征图（B, T, C', H', W'）
                output = fb_model(output1)  # 隐私分类：fb_model对特征图进行分类，输出logits（B, 1）
                 # 计算损失（BCEWithLogitsLoss，自动对logits应用sigmoid后计算二分类损失）
                loss = criterion(output,label)
                losses.append(loss.item())

            # 将预测结果（logits）和标签转移至CPU并转换为numpy格式
            predictions.extend(output.cpu().data.numpy()) # logits，后续可通过>0.5转为0/1
            vid_paths.extend(vid_path) # 收集视频路径（字符串列表）
            ground_truth.extend(label.cpu().data.numpy()) # 真实标签（0/1的float数组）

            if i+1 % 99 == 0: # 每99个批次打印中间结果（便于监控训练进度）
                print("Validation Epoch ", epoch, " Batch ", i, "- Loss : ", np.mean(losses))
        
    del inputs, output, label, loss # 释放内存，避免显存泄漏
    print("Validation Epoch ", epoch, "- Loss : ", np.mean(losses)) # 打印epoch平均损失

    ground_truth = np.asarray(ground_truth) # 将列表转为numpy数组（便于批量计算指标）
    # predictions = np.asarray(predictions)

    # 计算二分类指标：精确率、召回率、F1（宏平均，每个类别单独计算后取平均）
    prec, recall, f1, _ = precision_recall_fscore_support(ground_truth, (np.array(predictions) > 0.5).astype(int)) 
    predictions = np.asarray(predictions)
    ap = average_precision_score(ground_truth, predictions, average=None)  # 计算每个类别的平均精度（AP），二分类时返回两个值（对应0和1类）
    
    print(f'Macro f1 is {np.mean(f1)}') # 宏平均F1（二分类时为两个类别的平均）
    print(f'Macro prec is {np.mean(prec)}') # 宏平均精确率
    print(f'Macro recall is {np.mean(recall)}')  # 宏平均召回率
    print(f'Classwise AP is {ap}') # 每个类别的AP（二分类时[AP_0, AP_1]）
    print(f'Macro AP is {np.mean(ap)}') # 宏平均AP（所有类别的平均）

    # 按视频路径分组存储预测结果（同一视频可能有多个片段，后续可求平均）
    for entry in range(len(vid_paths)):
        if str(vid_paths[entry].split('/')[-1]) not in pred_dict.keys():
            pred_dict[str(vid_paths[entry].split('/')[-1])] = [] # 提取视频文件名（如vid.mp4）
            pred_dict[str(vid_paths[entry].split('/')[-1])].append(predictions[entry])

        else:
            # print('yes')
            pred_dict[str(vid_paths[entry].split('/')[-1])].append(predictions[entry])

    for entry in range(len(vid_paths)):
        if str(vid_paths[entry].split('/')[-1]) not in label_dict.keys():
            label_dict[str(vid_paths[entry].split('/')[-1])]= ground_truth[entry]

    return pred_dict, label_dict, np.mean(ap) # 返回分组预测、分组标签、宏平均AP

def train_classifier(run_id, restart, saved_model= None, kin_pretrained= True):  # kin_pretrained= False
    use_cuda = True
    best_score = 0
    writer = SummaryWriter(os.path.join(cfg.logs, str(run_id)))

    save_dir = os.path.join(cfg.saved_models_dir, run_id)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 在 train_classifier 函数中，save_dir 定义之后
    log_file = os.path.join(save_dir, 'metrics_log.txt')
    # 若文件不存在，写入表头
    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            f.write("epoch,privacy_AP,vad_AUC\n")

    ## 针对vad定义的参数
    # 在 train_classifier 函数开头或参数定义处
    vad_args = argparse.Namespace(
    method='pred',
    test_batch_size=1,
    h=params.height,
    w=params.width,
    t_length=params.t_length,
    alpha=0.6,
    th=0.01,
    num_workers_test=2
    )
    # 需要从 params 或外部配置读取测试集路径
    vad_test_folder = params.avenue_test_folder   
    vad_labels_path = params.avenue_labels_path

    ##写入txt日志
    def write_metrics_log(log_file_path, epoch, privacy_ap, vad_auc):
        """将 epoch 的指标写入 txt 文件（追加模式）"""
        with open(log_file_path, 'a') as f:
            f.write(f"{epoch},{privacy_ap:.6f},{vad_auc:.6f}\n")

    

    # 加载特征提取器（UNet，用于VISPR和UCF101的特征共享）
    fa_model = load_unet(saved_model_file = '/mnt/private/xwy/SPAct/initialization/ini_fa_avenue/model_103_bestAcc_0.0024.pth') #UNet(n_channels = 3, n_classes=3)
    
    # 初始化异常检测模型
    ft_model = convAE(params.channel, params.t_length, params.msize, params.fdim, params.mdim)
 
    # 初始化隐私分类器（如ResNet50，用于VISPR二分类）
    fb_model = build_privacy_classifier(arch = 'r50', num_classes = params.num_pa)

    # 学习率
    learning_rate_fa = params.learning_rate_fa
    learning_rate_fb = params.learning_rate_fb
    learning_rate_ft = params.learning_rate_ft

    # 损失函数
    criterion_ft = nn.MSELoss(reduction='none') # 均方误差损失
    criterion_fb = nn.BCEWithLogitsLoss() # 隐私分类损失（二分类BCEWithLogitsLoss，内部含sigmoid）
 
    # 初始化内存项（随机向量，归一化）
    m_items = F.normalize(torch.rand((params.msize, params.mdim), dtype=torch.float), dim=1).cuda() 
    
    # 移动模型至GPU 
    if torch.cuda.device_count()>1:
        print(f'Multiple GPUS found!')
        ft_model=nn.DataParallel(ft_model)
        fa_model=nn.DataParallel(fa_model)
        fb_model=nn.DataParallel(fb_model)
        ft_model.cuda()
        fa_model.cuda()
        fb_model.cuda()
    else:
        print('Only 1 GPU is available')
        ft_model.cuda()
        fa_model.cuda()
        fb_model.cuda()

    # 优化器
    optimizer_fa = optim.Adam(fa_model.parameters(),lr=params.learning_rate_fa)
    optimizer_fb = optim.Adam(fb_model.parameters(),lr=params.learning_rate_fb)
    optimizer_ft = optim.Adam(ft_model.parameters(),lr=params.learning_rate_ft)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_ft,T_max =params.num_epochs) # 余弦退火学习率调度器

    # avenue训练数据加载器（动作分类，强增强数据）
    train_dataset_avenue = vadDataLoader(params.avenue_train_folder, transforms.Compose([transforms.ToTensor(),]), resize_height=params.height, resize_width=params.width, time_step=params.t_length-1)
    # 分批次
    train_dataloader_avenue = DataLoader(train_dataset_avenue, batch_size=params.batch_size_avenue, shuffle=True, num_workers=params.num_workers, drop_last=True)
    print(f'avenue Train dataset length: {len(train_dataset_avenue)}')
    print(f'avenue Train dataset steps per epoch: {len(train_dataset_avenue)/params.batch_size_avenue}')

    # VISPR训练数据加载器（隐私分类，自监督帧对）
    train_dataset_vispr = vispr_ssl(datasplit = 'train', shuffle = False, data_percentage = params.data_percentage_vispr)
    # 分批次
    train_dataloader_vispr = DataLoader(train_dataset_vispr, batch_size=params.batch_size_vispr, shuffle=True, num_workers=params.num_workers, collate_fn=collate_fn_train)
    print(f'VISPR Train dataset length: {len(train_dataset_vispr)}')
    print(f'VISPR Train dataset steps per epoch: {len(train_dataset_vispr)/params.batch_size_vispr}')
   
    # 起始epoch
    epoch0  = 0
    val_array = [0, 3, 5,10,15,20, 25, 30, 35, 40, 45] + [50+ x for x in range(50)] # 定义验证epoch列表
    # val_array = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] # 目前先跑10轮测试

    # 生成数据增强参数组合（模式、跳帧、翻转、裁剪因子）
    modes = list(range(params.num_modes))
    skip = list(range(1,params.num_skips+1))
    hflip = params.hflip#list(range(2))
    cropping_fac1 = params.cropping_fac1#[0.7,0.85,0.8,0.75]
    print(f'Num modes {len(modes)}')
    print(f'Num skips {skip}')
    print(f'Cropping fac {cropping_fac1}')
    print(f'Base learning rate {params.learning_rate}')
    print(f'Scheduler patient {params.lr_patience}')
    print(f'Scheduler drop {params.scheduled_drop}')

    modes, skip,hflip, cropping_fac =  list(zip(*itertools.product(modes,skip,hflip,cropping_fac1)))
    accuracy = 0
    ft_w = params.ft_loss_weight
    # 主训练循环
    for epoch in range(epoch0, params.num_epochs):
        print(f'Epoch {epoch} started')
        start=time.time()
        # 核心训练函数：交替优化fa/fb/ft模型，返回更新后的模型
        fa_model, fb_model, ft_model, m_items ,ft_w= train_epoch(run_id, epoch, train_dataloader_vispr, train_dataloader_avenue, ft_model, fa_model, fb_model, m_items, criterion_ft, criterion_fb, optimizer_fa, optimizer_fb, optimizer_ft, writer, use_cuda,learning_rate_fa, learning_rate_fb, learning_rate_ft, ft_w)
        
        # 验证指定epoch
        if epoch in val_array:
            # VISPR验证：可视化fa_model的输入输出（仅第一个批次）
            validation_dataset = vispr_dataset_generator(datasplit = 'test', shuffle = False, data_percentage = params.data_percentage)
            validation_dataloader = DataLoader(validation_dataset, batch_size=10, shuffle=True, num_workers=params.num_workers, collate_fn=collate_fn_train_regular)     
            val_visualization_fa_vispr(save_dir, epoch, validation_dataloader, fa_model)  
            pred_dict, label_dict, macro_ap = val_epoch_vispr(run_id, epoch, validation_dataloader, fa_model, fb_model, criterion_fb, writer, use_cuda)                          
        # 新增 VAD 验证
        if hasattr(params, 'avenue_test_folder') and hasattr(params, 'avenue_labels_path'):
            auc_vad = val_epoch_vad(run_id, epoch, fa_model, ft_model, m_items,
                            params.avenue_test_folder, params.avenue_labels_path,
                            vad_args, use_cuda)
            writer.add_scalar('VAD_AUC', auc_vad, epoch)

        # ----- 写入日志文件 -----
        write_metrics_log(log_file, epoch, macro_ap, auc_vad)

        # 定期保存模型（每3个epoch）
        if epoch % 3 == 0:
            save_file_path = os.path.join(save_dir, 'model_{}.pth'.format(epoch))
            save_key_path = os.path.join(save_dir, 'keys_{}.pt'.format(epoch))
            states = {
                'epoch': epoch + 1,
                # 'arch': params.arch,
                'fa_model_state_dict': fa_model.state_dict(),
                'fb_model_state_dict': fb_model.state_dict(),
                'ft_model_state_dict': ft_model.state_dict(),
                'm_items': m_items  # 新增内存状态
            }
            torch.save(states, save_file_path)
            torch.save(fa_model, os.path.join(save_dir, 'fa_model_{}.pth'.format(epoch)))
            torch.save(ft_model, os.path.join(save_dir, 'ft_model_{}.pth'.format(epoch)))
            torch.save(m_items, save_key_path) # 保存内存模块记忆项
        # We will save optimizer weights for each temp model, not all saved models to reduce the storage
        
        # # 保存临时检查点（含优化器状态，用于断点续训） 
        save_file_path = os.path.join(save_dir, 'model_temp.pth')
        states = {
                'epoch': epoch + 1,
                # 'arch': params.arch,
                'fa_model_state_dict': fa_model.state_dict(),
                'fb_model_state_dict': fb_model.state_dict(),
                'ft_model_state_dict': ft_model.state_dict(),
                'optimizer_fa': optimizer_fa.state_dict(),
                'optimizer_fb': optimizer_fb.state_dict(),
                'optimizer_ft': optimizer_ft.state_dict(),
                'm_items': m_items  # 新增内存状态
            }
        torch.save(states, save_file_path)
        taken = time.time()-start
        # 保存视频异常检测的key
        torch.save(m_items, os.path.join(save_dir, 'keys.pt')) # 保存内存模块记忆项 

        print(f'Time taken for Epoch-{epoch} is {taken}')
        print()
        # 重新加载avenue（可能为了重新洗牌或更新数据增强）
        train_dataset_avenue = vadDataLoader(params.avenue_train_folder, transforms.Compose([transforms.ToTensor(),]), resize_height=params.height, resize_width=params.width, time_step=params.t_length-1)
        train_dataloader_avenue = DataLoader(train_dataset_avenue, batch_size=params.batch_size_avenue, shuffle=True, num_workers=params.num_workers, drop_last=True)
        print(f'avenue Train dataset length: {len(train_dataset_avenue)}')
        print(f'avenue Train dataset steps per epoch: {len(train_dataset_avenue)/params.batch_size_avenue}')

        # 重新加载vispr
        train_dataset_vispr = vispr_ssl(datasplit = 'train', shuffle = False, data_percentage = params.data_percentage_vispr)
        train_dataloader_vispr = DataLoader(train_dataset_vispr, batch_size=params.batch_size_vispr, shuffle=True, num_workers=params.num_workers, collate_fn=collate_fn_train)
        print(f'VISPR Train dataset length: {len(train_dataset_vispr)}')
        print(f'VISPR Train dataset steps per epoch: {len(train_dataset_vispr)/params.batch_size_vispr}')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Script to train baseline')

    parser.add_argument("--run_id", dest='run_id', type=str, required=False, default= "dummy_ssl_minmax",help='run_id')
    parser.add_argument("--restart", action='store_true')
    parser.add_argument("--kin_pretrained", action='store_true')

    args = parser.parse_args()
    print(f'Restart {args.restart}')
    print(f'kin_pretrained {args.kin_pretrained}')
    run_id = args.run_id

    train_classifier(str(run_id), args.restart, None, args.kin_pretrained)
