import sys
from pathlib import Path
from typing import Tuple

sys.path.append(r"C:\Users\Clavi\Desktop\coding\DANN-zcy")

import random
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torchvision import datasets, transforms
from dataset.data_loader import GetLoader
from models.model import CNNModel
import numpy as np
from train.test import test


def train() -> None:
    source_name = "MNIST"
    target_name = "mnist_m"

    proj_root = Path(__file__).resolve().parent.parent  # 项目根目录
    # 源域、目标域的数据路径
    source_root = proj_root / "dataset" / source_name
    target_root = proj_root / "dataset" / target_name
    
    # 保存模型的路径
    model_root = proj_root / "models"

    # 有 GPU 就加速，cudnn benchmark 自动选最优卷积算法
    cuda = True
    cudnn.benchmark = True

    # 超参：学习率、批大小、输入尺寸、训练轮数
    lr = 1e-3
    batch_size = 512
    image_size = 28
    n_epoch = 100

    # 固定随机种子，保证每次跑实验可复现
    manual_seed = random.randint(1, 10_000)
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    
    # ===================加载数据===================
    # 数据增强
    img_tf_source = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
        ]
    )

    img_tf_target = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )

    # --------------------源域--------------------
    # 数据读取（源域） 【训练自己的数据集需要修改】
    ds_source = datasets.MNIST(
        root=str(proj_root / "dataset"),
        train=True,
        transform=img_tf_source,
        download=False # 本地已有数据集，禁止重新下载
    )
    dl_source = torch.utils.data.DataLoader(
        dataset=ds_source, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=8 # 源域加载器，8 进程并行读图
    )

    # --------------------目标域--------------------
    # 目标域数据 mnist_m 标签文件：每行 "图片文件名 标签
    train_list = target_root / "mnist_m_train_labels.txt"
    
    # 目标域数据【训练自己的数据集需要修改】
    #　这里的GetLoader是自定义的类
    ds_target = GetLoader(
        data_root=str(target_root / "mnist_m_train"),
        data_list=str(train_list),
        transform=img_tf_target,
    )
    dl_target = torch.utils.data.DataLoader(
        dataset=ds_target, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=8
    )

    # ===================实例化模型、优化器、loss===================
    # load model
    # 实例化网络，里面包含：特征提取器 + 分类器 + 域判别器（含 GRL）
    net = CNNModel()

    # setup optimizer
    # 优化器： Adam 统一更新所有参数
    opt = optim.Adam(net.parameters(), lr=lr)

    # 故障分类器的损失、域分类器的损失都是负对数似然（网络最后会 log_softmax）
    loss_cls = torch.nn.NLLLoss()
    loss_dom = torch.nn.NLLLoss()

    if cuda: # 把模型和损失函数都搬去 GPU
        net = net.cuda()
        loss_cls = loss_cls.cuda()
        loss_dom = loss_dom.cuda()

    for p in net.parameters():
        # 告诉 PyTorch：这些张量 需要计算梯度（即会参与反向传播）。
        # 只有 requires_grad=True 的参数，优化器才会更新它们。
        # 但如果你在前面 冻结过（p.requires_grad=False）或者 加载了预训练模型、只想解冻继续训练，就显式把想训练的层重新打开。
        p.requires_grad = True

    # ===================开始训练===================
    for epoch in range(n_epoch):
        n_iter = min(len(dl_source), len(dl_target))
        iter_s = iter(dl_source)  # 转成迭代器，方便 next()
        iter_t = iter(dl_target)

        for i in range(n_iter):
            # 每个 iter 各拿一批源域和目标域

            # **梯度反转系数 α（GRL 核心）-----------------------------------------------
            # 控制 梯度反转层（GRL）的权重 α 随训练进度平滑地从 0 增长到 1。
            # 平滑 S 型，论文公式，反向传播时 乘在域判别器梯度上，实现“梯度反转”
            p = float(i + epoch * n_iter) / n_epoch / n_iter
            alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1

            # ---- source ----
            # 源域前向 + 误差
            s_img, s_label = next(iter_s)

            # 清空梯度
            net.zero_grad()
            bs = s_label.size(0)

            # 预分配张量，先占一块显存/内存，准备好一个指定形状的“空张量”，后面再用真实数据把它的值填进去。
            input_img = torch.empty(bs, 3, image_size, image_size)
            cls_label = torch.empty(bs, dtype=torch.long)
            # 源域域标签 = 0
            domain_label = torch.zeros(bs, dtype=torch.long)

            if cuda:
                s_img = s_img.cuda()
                s_label = s_label.cuda()
                input_img = input_img.cuda()
                cls_label = cls_label.cuda()
                domain_label = domain_label.cuda()

            # 把 1×28×28 广播复制到 3×28×28（三个通道一样）
            # 这两行就是把「真正数据」填到刚才 torch.empty 占好的坑里，完成一次「数据对齐 + 拷贝」。
            input_img.copy_(s_img.repeat(1, 3, 1, 1))
            cls_label.copy_(s_label)
            
            # **输入模型
            out_cls, out_dom = net(input_img, alpha=alpha)
            
            # 网络返回：分类 logits，域判别 logits
            err_s_cls = loss_cls(out_cls, cls_label)
            err_s_dom = loss_dom(out_dom, domain_label)

            # ---- target ----
            # 目标域前向 + 误差【目标域不训练故障分类器，只要域判别输出】
            t_img, _ = next(iter_t)
            
            bs = t_img.size(0)
            
            # 先占一块显存/内存
            input_img = torch.empty(bs, 3, image_size, image_size)
            
            # 目标域标签 = 1 
            domain_label = torch.ones(bs, dtype=torch.long)

            if cuda:
                t_img = t_img.cuda()
                input_img = input_img.cuda()
                domain_label = domain_label.cuda()

            input_img.copy_(t_img)

            _, out_dom = net(input_img, alpha=alpha)
            err_t_dom = loss_dom(out_dom, domain_label)

            # 总体误差
            loss = err_s_cls + err_s_dom + err_t_dom
            # 三股误差一起反向，Adam 更新特征提取器、分类器、域判别器全部参数
            loss.backward()
            opt.step()

            print(
                f"epoch: {epoch}, [iter: {i + 1} / {n_iter}], "
                f"err_s_label: {err_s_cls.item():.6f}, "
                f"err_s_domain: {err_s_dom.item():.6f}, "
                f"err_t_domain: {err_t_dom.item():.6f}"
            )

        torch.save(net, model_root / f"mnist_mnistm_model_epoch_{epoch}.pth")

        #-------一个epoch迭代完，算一次测试准确率---------
        test(source_name, epoch)
        test(target_name, epoch)

    print("done")


if __name__ == "__main__":
    train()