import numpy as np 
import torch.nn as nn
import torch
import torchvision
from torchsummary import summary
from r3d import r3d_18
from r3d_classifier import r3d_18_classifier
from r3d import mlp
from i3d import InceptionI3d
import parameters_BL as params


# from r2p1d import r2plus1d_18, embedder
# from classifier_r2p1d import r2plus1d_18_classifier
# # from try1_model import r2plus1d_18_changed
# # from dilated_r2plus1d import r2plus1d_18
from torchvision.models.utils import load_state_dict_from_url

def ssl_pretrained_load():
#     given = torch.load('/home/c3-0/ishan/privacy_preserving1/BYOL_load/model_final_checkpoint_phase799.torch')
    r50 = torchvision.models.resnet50(pretrained = False, progress = False)
    # pretrained = False：不加载在ImageNet 数据集上预训练的权重，得到一个随机初始化权重的 ResNet-50 模型。
    # progress = False：不显示下载预训练模型时的进度条      
# state_dict = r50.state_dict()
#     #The following is the backbone model loading

#     junk_trunk = '_feature_blocks.'                             
#     for key, value in given['classy_state_dict']['base_model']['model']['trunk'].items():
#         layer_name = key.replace('_feature_blocks.', '')
#         state_dict[layer_name] = value 
#     r50.load_state_dict(state_dict, strict = True)
    r50.fc = nn.Identity()  
    # 将全连接层替换为恒等映射层。Identity()是PyTorch中的一个模块，它的作用是直接返回输入的张量，不做任何改变。
    # 将ResNet-50作为一个特征提取器（而不是分类器），输出卷积层的特征
#     print('Backbone loaded successfully!')

    ############################ MLP building 定义一个简单的多层感知机，继承Module类

    class MLP(nn.Module):

        def __init__(self, final_embedding_size = 128, use_normalization = True):
        # 输出的特征向量维度128，归一化    
            super(MLP, self).__init__()

            self.final_embedding_size = final_embedding_size
            self.use_normalization = use_normalization
            # self.fc1 = nn.Linear(512*3*3,512)
            self.fc1 = nn.Linear(2048,2048, bias = True)  # 第一个全连接层 2048->2048
            self.relu = nn.ReLU(inplace=True)  # 激活函数 引入非线性
            self.fc2 = nn.Linear(2048, self.final_embedding_size, bias = True)  # 2048->128
        def forward(self, x):
        # 前向传播 归一化
            x = self.relu(self.fc1(x))
            #print(f'After fc1, shape of x is {x.shape}')
            x = nn.functional.normalize(self.fc2(x), p=2, dim=1)
            return x

    mlp = MLP()
    state_dict = mlp.state_dict()  # 获取模型参数 字典的键是参数名称 值是对应的参数张量
    # print(state_dict.keys())
    # exit()
    # given MLP ['0.clf.0.weight', '0.clf.0.bias', '1.clf.0.weight', '1.clf.0.bias']
    # my MLP has (['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias'])

    for key, value in given['classy_state_dict']['base_model']['model']['heads'].items():
        # print(key)
        layer_name = key.replace('0.clf.0', 'fc1')
        layer_name = layer_name.replace('1.clf.0', 'fc2')
        # print(layer_name)
        state_dict[layer_name] = value 
    mlp.load_state_dict(state_dict, strict = True)
    print('MLP loaded successfully!')
    combined = nn.Sequential(r50, mlp)  # 组合模型 数据首先通过r50，然后通过mlp
    return combined



def build_r3d_classifier(num_classes = 102, kin_pretrained = False, self_pretrained = True, saved_model_file = None):
    model = r3d_18_classifier(pretrained = kin_pretrained, progress = False)  # 基于R3D-18的分类器模型    
    model.layer4[0].conv1[0] = nn.Conv3d(256, 512, kernel_size=(3, 3, 3),\
                                stride=(1, 2, 2), padding=(2, 1, 1),dilation = (2,1,1), bias=False)
    # 修改某些参数
    model.layer4[0].downsample[0] = nn.Conv3d(256, 512,\
                          kernel_size = (1, 1, 1), stride = (1, 2, 2), bias=False)

    if self_pretrained == True:
        pretrained = torch.load(saved_model_file)  # 加载指定的预训练模型权重
        pretrained_kvpair = pretrained['state_dict']
        # print(pretrained_kvpair)
        # exit()
        # 对预训练权重的键名进行处理，使其与当前模型的键名相匹配
        model_kvpair = model.state_dict()
        for layer_name, weights in pretrained_kvpair.items():
            if 'module.1.' in layer_name:
                continue
            elif layer_name[:2] == '1.':
                continue
            if layer_name[:2] != '0.':
                layer_name = layer_name.replace('module.0.','')
            else:
                layer_name = layer_name[2:]

            model_kvpair[layer_name] = weights   
        model.load_state_dict(model_kvpair, strict=True)  # 将处理后的预训练权重加载到模型中
        print(f'model {saved_model_file} loaded successsfully!')
    # exit()
    model.fc = nn.Linear(512, num_classes)  # 修改全连接层，将其替换为线性层，510->102
    return model 

def load_r3d_classifier(num_classes = 102, saved_model_file = None):
    model = r3d_18_classifier(pretrained = False, progress = False)

    model.layer4[0].conv1[0] = nn.Conv3d(256, 512, kernel_size=(3, 3, 3),\
                                stride=(1, 2, 2), padding=(2, 1, 1),dilation = (2,1,1), bias=False)
    model.layer4[0].downsample[0] = nn.Conv3d(256, 512,\
                          kernel_size = (1, 1, 1), stride = (1, 2, 2), bias=False)

    model.fc = nn.Linear(512, num_classes)

    pretrained = torch.load(saved_model_file)  # 指定路径下的预训练权重文件
    pretrained_kvpair = pretrained['state_dict']
    # print(pretrained_kvpair)
    # exit()

    model_kvpair = model.state_dict()
    for layer_name, weights in pretrained_kvpair.items():
        layer_name = layer_name.replace('module.0.','')
        model_kvpair[layer_name] = weights   
    model.load_state_dict(model_kvpair, strict=True)
    print(f'model {saved_model_file} loaded successsfully!')
    return model 

# 构建修改后的r3d骨干网络，主要用于特征提取
def build_r3d_backbone():
    model = r3d_18(pretrained = False, progress = False)
    
    model.layer4[0].conv1[0] = nn.Conv3d(256, 512, kernel_size=(3, 3, 3),\
                                stride=(1, 2, 2), padding=(2, 1, 1),dilation = (2,1,1), bias=False)
    model.layer4[0].downsample[0] = nn.Conv3d(256, 512,\
                          kernel_size = (1, 1, 1), stride = (1, 2, 2), bias=False)
    return model

# 构建一个包含r3d骨干网络和mlp的组合模型
def build_r3d_mlp():
    f = build_r3d_backbone()
    g = mlp()
    model = nn.Sequential(f,g) # 先经过r3d再经过mlp
    return model
    
def load_r3d_mlp(saved_model_file):
    model = build_r3d_mlp()  # 构建组合模型
    pretrained = torch.load(saved_model_file)  # 加载预训练权重
    pretrained_kvpair = pretrained['state_dict']
    model_kvpair = model.state_dict()

    for layer_name, weights in pretrained_kvpair.items():
        # if 'module.1' in layer_name: # removing embedder part which is module.1 in the model+embedder
        #     continue
        layer_name = layer_name.replace('module.','')
        model_kvpair[layer_name] = weights  

    model.load_state_dict(model_kvpair, strict=True)
    print(f'{saved_model_file} loaded successfully')
    
    return model 

# 构建原始i3d模型 基于i3d结构进行分类
def build_i3dori_classifier(num_classes =401, dropout_keep_prob = 0.5):
    model = InceptionI3d(num_classes = num_classes, dropout_keep_prob = dropout_keep_prob)
    return model

#This file is just for model loading
def model_building(restart, save_dir, saved_model, architecture):
    '''
    如果restart为True,首先尝试加载指定路径的模型权重，
    如果模型不存在,根据saved_model是否为空来决定构建新模型还是进行基线实验模型的构建。
    如果saved_model不为空,则尝试加载其指定的预训练模型。

    如果restart为False,同样根据saved_model是否为空以及指定的架构,选择构建新模型或加载相应的预训练模型。

    epoch:记录模型训练轮次 方便训练恢复、训练进度记录、实验对比和复现
    '''
    if restart:
            saved_model_file = save_dir + '/model_temp.pth'
            
            if os.path.exists(saved_model_file):
                model = load_r3d_classifier(saved_model_file= saved_model_file)
                epoch0 = torch.load(saved_model_file)['epoch']
            else:
                print(f'No such model exists: {saved_model_file} :(')
                if not (saved_model == None or len(saved_model) == 0):
                    print(f'Trying to load {saved_model[30:]}')
                    model = build_r3d_classifier(saved_model_file = saved_model[30:], num_classes = params.num_classes)
                else:
                    print(f'It`s a baseline experiment!')
                    model = build_r3d_classifier(self_pretrained = False, saved_model_file = None, num_classes = params.num_classes) 
                epoch0 = 0
    
    else:

        if not (saved_model == None or len(saved_model) == 0):
            print(f'Trying to load {saved_model[30:]}')
            if architecture == 'r3d18':
                model = build_r3d_classifier(saved_model_file = saved_model[30:], num_classes = params.num_classes)
            elif architecture == 'i3dori':
                model = build_i3dori_classifier(num_classes = params.num_classes)
        else:
            print(f'It`s a baseline experiment!')
            if architecture == 'r3d18':
                model = build_r3d_classifier(self_pretrained = False, saved_model_file = None, num_classes = params.num_classes)
            elif architecture == 'i3dori':
                model = build_i3dori_classifier(num_classes = params.num_classes)
        epoch0 = 0
    return model, epoch0

if __name__ == '__main__':
    # model = arch_r50_classifier(num_classes = 101)
    # model.eval()
    # model.cuda()
    # summary(model, (16, 3, 112, 112))
    input = torch.rand(5, 3, 16, 112, 112).cuda()
    # model = build_r3d_classifier(num_classes = 102, saved_model_file = '/home/c3-0/ishan/ss_saved_models/r3d59v2cont2/model_best_e89_loss_15.573.pth')
    model = build_r3d_classifier(num_classes = 102, saved_model_file = '/home/c3-0/ishan/ss_saved_models/r3d59v2cont2/model_best_e102_loss_15.418.pth')

    # sftp://ishan@crcv.eecs.ucf.edu/home/c3-0/ishan/ss_saved_models/r3d59v2cont2/model_best_e102_loss_15.418.pth
    # model = torchvision.models.video.r2plus1d_18(pretrained=True, progress=True)

    # saved_model_file = '/home/c3-0/ishan/ss_saved_models/4self_1e-3/model_e108_loss_0.00.pth'
    # model1 = load_r2plus1d_18_dilatedv2(saved_model_file= saved_model_file)
    # model = nn.Sequential(*list(model.children()))
    # print(model)
    # print()
    # model = r2plus1d_18_dilated()
    # print(model)

    # # print()
    model.eval()
    model.cuda()
    output = model(input)
    print(input.shape)
    print(output.shape)
    # model1.eval()
    # model1.cuda()
    # output = model((input,'s'))

    # print(output[0].shape)
    # summary(model1, (3, 16, 112, 112))
    # summary(model, (1,(3, 16, 112, 112)))

