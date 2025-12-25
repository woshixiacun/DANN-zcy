import sys
# 把项目根目录临时塞进 Python 搜索路径，保证下面 from dataset / from models 能 import 到
sys.path.append(r"C:\Users\Clavi\Desktop\coding\DANN-zcy")

import random
import os
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from dataset.data_loader import GetLoader
from torchvision import datasets
from torchvision import transforms
from models.model import CNNModel
import numpy as np
from train.test import test

def train():

    #源域、目标域的数据路径
    source_dataset_name = 'MNIST'
    target_dataset_name = 'mnist_m'
    source_image_root = os.path.join(r"C:\Users\Clavi\Desktop\coding\DANN-zcy", 'dataset', source_dataset_name)
    target_image_root = os.path.join(r"C:\Users\Clavi\Desktop\coding\DANN-zcy", 'dataset', target_dataset_name)
   
   #保存模型的路径
    model_root = os.path.join('..', 'models')
   
   # 有 GPU 就加速，cudnn benchmark 自动选最优卷积算法
    cuda = True
    cudnn.benchmark = True

    # 超参：学习率、批大小、输入尺寸、训练轮数
    lr = 1e-3
    # batch_size = 128
    batch_size = 512
    image_size = 28
    n_epoch = 100

    # 固定随机种子，保证每次跑实验可复现
    manual_seed = random.randint(1, 10000)
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    # load data
    # 数据增强
    img_transform_source = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])

    img_transform_target = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) # # MNIST-M 是 3 通道，用 0.5 方便把像素归到 [-1,1]
    ])

    # 数据读取（源域） 【训练自己的数据集需要修改】
    dataset_source = datasets.MNIST(
        # root='../dataset',
        root='C:/Users/Clavi/Desktop/coding/DANN-zcy/dataset',
        train=True,
        transform=img_transform_source,
        # download=True
        download=False  # 本地已有数据集，禁止重新下载
    )

    dataloader_source = torch.utils.data.DataLoader(
        dataset=dataset_source,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8)  # 源域加载器，8 进程并行读图

    # 源域数据 MNIST-M 标签文件：每行 "图片文件名 标签
    train_list = os.path.join(target_image_root, 'mnist_m_train_labels.txt')
    
    # 目标域数据【训练自己的数据集需要修改】
    dataset_target = GetLoader(
        data_root=os.path.join(target_image_root, 'mnist_m_train'),
        data_list=train_list,
        transform=img_transform_target
    )

    dataloader_target = torch.utils.data.DataLoader(
        dataset=dataset_target,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8)

    # load model
    # 实例化网络，里面包含：特征提取器 + 分类器 + 域判别器（含 GRL）
    my_net = CNNModel()

    # setup optimizer
    # 优化器： Adam 统一更新所有参数
    optimizer = optim.Adam(my_net.parameters(), lr=lr)

    # 故障分类器的损失、域分类器的损失都是负对数似然（网络最后会 log_softmax）
    loss_class = torch.nn.NLLLoss()
    loss_domain = torch.nn.NLLLoss()

    if cuda:
        # 把模型和损失函数都搬去 GPU
        my_net = my_net.cuda()
        loss_class = loss_class.cuda()
        loss_domain = loss_domain.cuda()

    for p in my_net.parameters():
        # 告诉 PyTorch：这些张量 需要计算梯度（即会参与反向传播）。
        # 只有 requires_grad=True 的参数，优化器才会更新它们。
        # 但如果你在前面 冻结过（p.requires_grad=False）或者 加载了预训练模型、只想解冻继续训练，就显式把想训练的层重新打开。
        p.requires_grad = True

    # training
    # 训练大循环：epoch
    for epoch in range(n_epoch):

        len_dataloader = min(len(dataloader_source), len(dataloader_target)) #取短的一方，避免某一方提前耗尽
        data_source_iter = iter(dataloader_source)  # 转成迭代器，方便 next()
        data_target_iter = iter(dataloader_target)

        i = 0
        while i < len_dataloader:
            # # 每个 iter 各拿一批源域和目标域

            # **梯度反转系数 α（GRL 核心）-----------------------------------------------
            # 控制 梯度反转层（GRL）的权重 α 随训练进度平滑地从 0 增长到 1。
            # 平滑 S 型，论文公式，反向传播时 乘在域判别器梯度上，实现“梯度反转”
            p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # training model using source data
            # 源域前向 + 误差-----------------------------------------------
            data_source = data_source_iter.next()

            s_img, s_label = data_source

            # 清空梯度33
            my_net.zero_grad()
            batch_size = len(s_label)

            # 预分配张量，下面把灰度图复制成 3 通道
            input_img = torch.FloatTensor(batch_size, 3, image_size, image_size)
            class_label = torch.LongTensor(batch_size)
            # 源域域标签 = 0
            domain_label = torch.zeros(batch_size)
            domain_label = domain_label.long()

            if cuda:  #全部搬到 GPU
                s_img = s_img.cuda()
                s_label = s_label.cuda()
                input_img = input_img.cuda()
                class_label = class_label.cuda()
                domain_label = domain_label.cuda()

            # 把 1×28×28 广播复制到 3×28×28（三个通道一样）
            input_img.resize_as_(s_img).copy_(s_img)
            class_label.resize_as_(s_label).copy_(s_label)

            # **输入模型
            class_output, domain_output = my_net(input_data=input_img, alpha=alpha)

            # 网络返回：分类 logits，域判别 logits
            err_s_label = loss_class(class_output, class_label)
            err_s_domain = loss_domain(domain_output, domain_label)
            
            # training model using target data
            # 目标域前向 + 误差-----------------------------------------------
            # 目标域 不训练故障分类器，只要域判别输出
            data_target = data_target_iter.next()
            t_img, _ = data_target

            batch_size = len(t_img)

            input_img = torch.FloatTensor(batch_size, 3, image_size, image_size)

            # 目标域标签 = 1 
            domain_label = torch.ones(batch_size)
            domain_label = domain_label.long()

            if cuda:
                t_img = t_img.cuda()
                input_img = input_img.cuda()
                domain_label = domain_label.cuda()

            input_img.resize_as_(t_img).copy_(t_img)

            _, domain_output = my_net(input_data=input_img, alpha=alpha)

            err_t_domain = loss_domain(domain_output, domain_label)
            # 总体误差
            err = err_t_domain + err_s_domain + err_s_label
            # 三股误差一起反向，Adam 更新特征提取器、分类器、域判别器全部参数
            err.backward()
            optimizer.step()

            i += 1

            print ('epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_domain: %f' \
                % (epoch, i, len_dataloader, err_s_label.cpu().data.numpy(),
                    err_s_domain.cpu().data.numpy(), err_t_domain.cpu().data.numpy()))

        torch.save(my_net, '{0}/mnist_mnistm_model_epoch_{1}.pth'.format(model_root, epoch))
        test(source_dataset_name, epoch)
        test(target_dataset_name, epoch)

    print( 'done')



if __name__ == '__main__':

    train()