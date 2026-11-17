import torch
# from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import *
import time
import os
import numpy as np
# from model import build_r3d_classifier, load_r3d_classifier
import parameters_BL as params
import config as cfg
from dl_reconstruction import *
import sys, traceback
# from sklearn.metrics import precision_recall_fscore_support, average_precision_score
from tensorboardX import SummaryWriter
# import cv2
from torch.utils.data import DataLoader
import math
import argparse
import itertools
from unet_model import UNet
import warnings
from torchvision.utils import save_image

warnings.filterwarnings("ignore", category=DeprecationWarning) 

from keras.utils import to_categorical

# if torch.cuda.is_available(): 
#     torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.backends.cudnn.benchmark = True

# 训练一个epoch的函数，用于执行模型的训练步骤，包括前向传播、计算损失、反向传播和参数更新，并记录训练损失。每500个batch打印一次当前的平均损失
def train_epoch(run_id, epoch, data_loader, fa_model, criterion, optimizer, writer, use_cuda,learning_rate2):
# run_id：运行的唯一标识符，用于保存训练记录和模型文件。
# epoch：当前训练的轮数。只作打印用
# data_loader：训练数据的 DataLoader，用于提供批量的训练数据。
# fa_model：要训练的模型。
# criterion：损失函数，用于计算模型输出与真实标签之间的差异。
# optimizer：优化器，用于更新模型的参数。
# writer：SummaryWriter 对象，用于记录训练过程中的数据，如损失、学习率等，以便在 TensorBoard 中可视化。
# use_cuda：布尔值，表示是否使用 CUDA 加速。
# learning_rate2：当前的学习率。

    print('train at epoch {}'.format(epoch))  # 打印当前训练周期
    for param_group in optimizer.param_groups:  # 更新优化器的学习率，将学习率设置为learning_rate2
        param_group['lr']=learning_rate2
        writer.add_scalar('Learning Rate', learning_rate2, epoch)  #将当前学习率记录在TensorBoard中

        print("Learning rate is: {}".format(param_group['lr']))  # 打印当前学习率
  
    losses, weighted_losses = [], []  # 存储每个批次的损失值
    # loss_mini_batch = 0
    # optimizer.zero_grad()

    fa_model.train()  # 设置模型为训练模式

    for i, (inputs, vid_path) in enumerate(data_loader):
        # print(f'label is {label}')
        # inputs = inputs.permute(0,4,1,2,3)
        # print(inputs.shape, flush= True)
        optimizer.zero_grad()  # 清空之前计算的梯度，避免梯度累积

        # inputs = inputs.permute(0,2,1,3,4)
        # print(inputs.shape)
        
        # inputs = inputs.permute(0,2,1,3,4)
        if use_cuda:
            inputs = inputs.cuda()  # 将数据移动到GPU上
        
        # print(inputs.shape)
        output = fa_model(inputs) # 获得模型输出
        loss = criterion(output,inputs)  # 计算模型输出与输入数据之间的损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新模型参数
        losses.append(loss.item())  # 将当前的损失值添加到损失表中
        if i % 500 == 0: 
            print(f'Training Epoch {epoch}, Batch {i}, Loss: {np.mean(losses) :.5f}', flush = True) # 输出当前epoch 批次索引 累计平均损失
        
    print('Training Epoch: %d, Loss: %.4f' % (epoch,  np.mean(losses)))  # 打印epoch和平均损失
    writer.add_scalar('Training Loss', np.mean(losses), epoch) # 将当前epoch的平均损失记录到TensorBoard中

    del loss, inputs, output  # 删除不再使用的变量，释放内存

    return fa_model, np.mean(losses)  # 返回训练后的模型和当前epoch的平均损失

# validation_epoch(run_id, epoch, train_dataloader, fa_model,  criterion, optimizer, writer, use_cuda, learning_rate2)
# def val_epoch(run_id, epoch,mode, skip, hflip, cropping_fac, pred_dict,label_dict, data_loader, fa_model,  criterion, writer, use_cuda):
    
def validation_epoch(run_id, epoch, data_loader, fa_model,  criterion, optimizer, writer, use_cuda, learning_rate2):
    # run_id：当前训练任务的唯一标识符，用于保存验证过程中生成的可视化图像。
    # epoch：当前训练轮次，用于文件名和日志记录。
    # data_loader：验证数据集的 DataLoader，提供批量验证数据。
    # fa_model：待验证的模型（如 U-Net 重构模型）。
    # criterion：损失函数（与训练时一致，此处为 L1 损失）。
    # optimizer/writer/use_cuda/learning_rate2：沿用训练时的配置，其中 writer 用于记录验证损失到 TensorBoard。
    
    fa_model.eval()  # 设置模型为评估模式

    losses = []
    # predictions, ground_truth = [], []
    vid_paths = []
    save_dir = os.path.join(cfg.saved_models_dir, run_id)  # 保存验证过程的图像
    for i, (inputs, vid_path) in enumerate(data_loader):
        vid_paths.extend(vid_path)
        # inputs = inputs.permute(0,4,1,2,3)
        if len(inputs.shape) != 1:

            # inputs = inputs.permute(0, 2, 1, 3, 4)

            if use_cuda:
                inputs = inputs.cuda()
            
            # print(inputs.shape)
            with torch.no_grad():
        
            
                output = fa_model(inputs)
                
                loss = criterion(output,inputs)
                losses.append(loss.item())

            # print(len(predictions))

        if i % 300 == 0:   # 每300个批次，打印当前验证周期、批次和平均损失，并从输入和输出中各取前5个样本，将它们拼接在一起并保存为图像。
            print(f'Validation Epoch {epoch}, Batch {i}, Loss: {np.mean(losses) :.5f}', flush = True)
            vis_image = torch.cat([inputs[:5], output[:5]], dim=0)

            save_image(vis_image, save_dir+'/combined'+'_epoch_'+ str(epoch) + '_batch_'+ str(i)  + '.png', padding=5, nrow=5)  # 保存拼接后的可视化图像

    del inputs, output, loss 
    
    print('Validation Epoch: %d, Loss: %.4f' % (epoch,  np.mean(losses)))
    writer.add_scalar('Validation Loss', np.mean(losses), epoch)
    return np.mean(losses)
    
def train_classifier(run_id, restart, saved_model):  # 整个训练流程的入口 定义了训练分类器
    # run_id（str）：当前训练任务的唯一标识符，用于区分不同训练实例，生成日志和模型保存路径。
    # restart（bool）：是否重启训练（若为 True，则忽略 saved_model，从头开始训练）。
    # saved_model（str）：可选的预训练模型路径，用于加载已保存的模型参数和优化器状态，继续训练。
    use_cuda = True
    best_score = 10000 # 初始化最佳得分，用于比较验证损失，保存最佳模型
    writer = SummaryWriter(os.path.join(cfg.logs, str(run_id)))  # 创建一个SummaryWriter对象，用于记录训练过程中的数据，以便在TensorBoard中可视化

    save_dir = os.path.join(cfg.saved_models_dir, run_id)  # 构建保存模型的目录路径
    if not os.path.exists(save_dir):  # 如果保存的目录不存在，则创建该目录
        os.makedirs(save_dir)
    
    fa_model = UNet(n_channels = 3, n_classes=3)  # 创建一个unet模型实例，输入通道为3，输出类别数为3
    epoch0 = 0
   

    # saved_model_file = '/home/c3-0/ishan/ss_saved_models/r3d58/model_best_e154_loss_0.7567.pth'
    learning_rate1 = params.learning_rate  # 从parameters_BL模块中获取学习率参数
    
    # temp = list(np.linspace(0,1, 10) + 1e-9) + [1 for i in range(50)] + [0.1 for i in range(100)]

    # lr_array = l*np.asarray(temp)
    learning_rate2 = learning_rate1  # 初始化学习率为基础学习率

    
    # print(lr_array[:50])
    criterion= nn.L1Loss()  # 创建一个L1损失函数实例，计算输入张量和目标张量之间的绝对差值的平均值

    if torch.cuda.device_count()>1:  # 看有几个GPU可用
        print(f'Multiple GPUS found!')
        fa_model=nn.DataParallel(fa_model)  # 如果有多个GPU可用，使用DataParallel对模型进行并行化处理，加速训练
        criterion.cuda()  # 将损失函数移动到GPU上
        fa_model.cuda()  # 将模型移动到GPU上
    else: 
        print('Only 1 GPU is available')
        criterion.cuda()
        fa_model.cuda()


    optimizer = optim.Adam(fa_model.parameters(),lr=params.learning_rate)  # 创建一个Adam优化器实例，用于更新模型的参数
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.80)
    train_dataset = reconstruction_dataloader(datasplit = 'train', shuffle = False, data_percentage = 1.0)  #加载训练数据集

    train_dataloader = DataLoader(train_dataset, batch_size=params.batch_size, \
        shuffle=True, num_workers=params.num_workers, collate_fn=collate_fn_train)  # 创建一个DataLoader对象，用于批量加载训练数据
    print(f'Train dataset length: {len(train_dataset)}')  #打印数据集的长度
    print(f'Train dataset steps per epoch: {len(train_dataset)/params.batch_size}')  # 打印每个epoch的训练步数
    # val_array = list(range(250))#[0,1,25,50,100,150,175] + list(range(200,255)) 
    # val_array = [0,9,60] + [90, 93, 96, 99] + [125+ x for x in range(100)]
    val_array = [0,3,5,10,12,15, 20, 25, 30, 35, 40, 45] + [50+ x for x in range(100)]  # 指定在哪些epoch进行验证


    modes = list(range(params.num_modes))  # 获取模式数量的范围列表
    skip = list(range(1,params.num_skips+1))  # 获取跳跃步长的范围列表
    hflip = params.hflip#list(range(2))  # 获取水平翻转参数
    cropping_fac1 = params.cropping_fac1#[0.7,0.85,0.8,0.75]  # 获取裁剪因子列表
    print(f'Num modes {len(modes)}')  # 打印模式数量
    print(f'Num skips {skip}')  # 打印跳跃步长数量
    print(f'Cropping fac {cropping_fac1}')  # 打印裁剪因子
    print(f'Base learning rate {params.learning_rate}')  # 打印基础学习率
    print(f'Scheduler patient {params.lr_patience}')  # 打印学习率调度器的耐心值
    print(f'Scheduler drop {params.scheduled_drop}')  # 打印学习率调度器的下降因子

    modes, skip,hflip, cropping_fac =  list(zip(*itertools.product(modes,skip,hflip,cropping_fac1)))   # 使用itertools.product生成模式、跳跃步长、水平翻转和裁剪因子的所有组合
    accuracy = 0
    lr_flag1 = 0
    lr_counter = 0
    # train_loss_prev = 1000
    # lr_array = [0.001, 0.01, 0.1, 1] + [1 for x in range(15)] + [0.5 for x in range(15)] +  [0.1 for x in range(30)] + [0.05 for x in range(20)] + [0.01 for x in range(25)]
    # lr_array = np.asarray(lr_array)*learning_rate1
    learning_rate1 = params.learning_rate # 设置学习率为参数配置中的初始学习率
    train_loss_best = 1000  # 初始化最佳训练损失

    for epoch in range(epoch0, params.num_epochs):
        if epoch < params.warmup and lr_flag1 ==0:
            learning_rate2 = params.warmup_array[epoch]*params.learning_rate
        # learning_rate2 = lr_array[epoch]

        print(f'Epoch {epoch} started')
        start=time.time()
        try:
            # 调用train_epoch函数进行一个epoch的训练，并获取训练后的模型和训练损失
            fa_model, train_loss = train_epoch(run_id, epoch, train_dataloader, fa_model,  criterion, optimizer, writer, use_cuda, learning_rate2)
            
            # if train_loss > train_loss_prev:
            #     lr_counter += 1
            
            # if lr_counter > params.lr_patience:
            #     lr_counter = 0
            #     learning_rate2 = learning_rate2/params.scheduled_drop
            #     print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            #     print(f'Learning rate dropping to its {scheduled_drop}th value to {learning_rate2} at epoch {epoch}')
            #     print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

            # train_loss_prev = train_loss
            # 根据训练损失调整学习率
            if train_loss > train_loss_best:
                lr_counter += 1
            if lr_counter > params.lr_patience:
                lr_counter = 0
                learning_rate2 = learning_rate2/params.scheduled_drop
                print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                print(f'Learning rate dropping to its {params.scheduled_drop}th value to {learning_rate2} at epoch {epoch}')
                print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')



            if train_loss_best < train_loss:
                train_loss_best = train_loss

            # if train_loss < 0.8 and lr_flag1 ==0:
            #     lr_flag1 =1 
            #     learning_rate2 = learning_rate1/2
            #     print(f'Dropping learning rate to {learning_rate2} for epoch')

            # if train_loss < 0.4 and lr_flag1 ==1:
            #     lr_flag1 = 2  
            #     learning_rate2 = learning_rate1/10
            #     print(f'Dropping learning rate to {learning_rate2} for epoch')

            # if train_loss < 0.25 and lr_flag1 ==2:
            #     lr_flag1 = 3
            #     learning_rate2 = learning_rate1/20
            #     print(f'Dropping learning rate to {learning_rate2} for epoch')

            # if train_loss < 0.20 and lr_flag1 ==3:
            #     lr_flag1 = 4
            #     learning_rate2 = learning_rate1/100
            #     print(f'Dropping learning rate to {learning_rate2} for epoch')

            # if train_loss < 0.05 and lr_flag1 ==4:
            #     lr_flag1 = 5
            #     learning_rate2 = learning_rate1/1000
            #     print(f'Dropping learning rate to {learning_rate2} for epoch')

            # 若当前epoch在之前指定的epoch验证列表中时
            if epoch in val_array:
                # 加载测试数据集
                validation_dataset = reconstruction_dataloader(datasplit = 'test', shuffle = False, data_percentage = 1.0)

                validation_dataloader = DataLoader(validation_dataset, batch_size=params.batch_size, \
                shuffle=True, num_workers=params.num_workers, collate_fn=collate_fn_train)

                val_loss = validation_epoch(run_id, epoch, train_dataloader, fa_model,  criterion, optimizer, writer, use_cuda, learning_rate2)  # 获取验证损失 

            # 记录最佳模型
            if val_loss < best_score:
                print('++++++++++++++++++++++++++++++')
                print(f'Epoch {epoch} is the best model till now for {run_id}!')
                print('++++++++++++++++++++++++++++++')
                # 保存路径
                save_dir = os.path.join(cfg.saved_models_dir, run_id) 
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                # 生成文件名 文件名格式：model_<epoch>_bestAcc_<val_loss前6位>.pth 示例：model_50_bestAcc_0.3215.pth
                save_file_path = os.path.join(save_dir, 'model_{}_bestAcc_{}.pth'.format(epoch, str(val_loss)[:6]))

                # 构建保存状态字典
                states = {
                    'epoch': epoch + 1, # 下一个epoch的起始值
                    'lr_counter' : lr_counter, # 学习率调整计数器
                    # 'arch': params.arch,
                    'fa_model_state_dict': fa_model.state_dict(), # 模型参数（权重和偏置）
                    'optimizer': optimizer.state_dict(), # 优化器状态（如 Adam 的动量、学习率历史）
                }
                torch.save(states, save_file_path) # 保存模型状态文件 将 states 字典序列化并保存为 .pth 文件，支持后续通过 torch.load() 加载。
                best_score = val_loss # 更新最佳损失
            # 保存临时模型
            save_dir = os.path.join(cfg.saved_models_dir, run_id)
            save_file_path = os.path.join(save_dir, 'model_temp.pth')
            states = {
                'epoch': epoch + 1,
                'lr_counter' : lr_counter,
                # 'arch': params.arch,
                'fa_model_state_dict': fa_model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(states, save_file_path)
            # scheduler.step()
            # elif epoch % 20 == 0:
            #     save_dir = os.path.join(cfg.saved_models_dir, run_id)
            #     save_file_path = os.path.join(save_dir, 'model_{}_Acc_{}_F1_{}.pth'.format(epoch, str(accuracy)[:6],str(f1_score)[:6]))
            #     states = {
            #         'epoch': epoch + 1,
            #         # 'arch': params.arch,
            #         'state_dict': model.state_dict(),
            #         'optimizer': optimizer.state_dict(),
            #     }
            #     torch.save(states, save_file_path)

            # scheduler.step()
        except:
            print("Epoch ", epoch, " failed")
            print('-'*60)
            traceback.print_exc(file=sys.stdout)
            print('-'*60)
            continue
        # 监控训练耗时、重新初始化训练数据加载器并打印数据集信息
        # 计算并打印epoch耗时
        taken = time.time()-start
        print(f'Time taken for Epoch-{epoch} is {taken}')
        print()
        # 重新加载训练数据集 指定训练加载集 数据集本身不打乱顺序（但后续 DataLoader 会通过 shuffle=True 打乱） 加载全部数据（非子集）
        train_dataset = reconstruction_dataloader(datasplit = 'train', shuffle = False, data_percentage = 1.0)
        # 重新加载训练数据加载器
        train_dataloader = DataLoader(train_dataset, batch_size=params.batch_size, \
        shuffle=True, num_workers=params.num_workers, collate_fn=collate_fn_train)
        # 打印数据集信息
        print(f'Train dataset length: {len(train_dataset)}')
        print(f'Train dataset steps per epoch: {len(train_dataset)/params.batch_size}')


        

if __name__ == '__main__':
    # 获取从终端输入的参数
    parser = argparse.ArgumentParser(description='Script to train baseline') # 参数解析器

    parser.add_argument("--run_id", dest='run_id', type=str, required=False, default= "dummy_recon",
                        help='run_id')
    parser.add_argument("--restart", action='store_true')
    parser.add_argument("--saved_model", dest='saved_model', type=str, required=False, default= None,
                        help='run_id')

    # print()
    # print('Repeating r3d57, Optimizer grad inside each iteration')
    # print()

    args = parser.parse_args()
    print(f'Restart {args.restart}')

    run_id = args.run_id
    saved_model = args.saved_model

    train_classifier(str(run_id), args.restart, saved_model)


        


