import sys
from pathlib import Path
from typing import Tuple
ROOT = Path(__file__).resolve().parent.parent  # 项目根目录
sys.path.append(str(ROOT))

import random
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torchvision import datasets, transforms
from dataset.data_loader import GetLoader
from dataset.data_loader_adopt import GetLoader_L

from models.model1D import CNNModel_1D
from models.functions import mkglmfa_rkhs_loss

import numpy as np
from train.test_new import test_new



def train() -> None:
    # ===================参数/路径/随机种子初始化===================
    source_name = "1Ddata_convet_zcy"
    target_name = "1Ddata_convet_zcy"

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
    batch_size = 8
    image_size = 28
    n_epoch = 100

    # 固定随机种子，保证每次跑实验可复现
    manual_seed = 0    # random.randint(1, 10_000)
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)

    # ===================构造 MNIST与 MNIST-M的DataLoader===================
    # 数据增强
    img_tf_source = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    img_tf_target = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    # --------------------源域--------------------
    # 数据读取（源域） 【训练自己的数据集需要修改】

    datasets_source = GetLoader_L(data_root=source_root,
                                  data_list="train.pkl",
                                  transform=None) 
    
    dataloader_source = torch.utils.data.DataLoader(
        dataset=datasets_source,
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0
    )

    # --------------------目标域--------------------


    datasets_target = GetLoader_L(data_root=source_root,
                                  data_list="train.pkl",
                                  transform=None) 
    
    dataloader_target = torch.utils.data.DataLoader(
        dataset=datasets_target,
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0
    )

    # ===================实例化CNN模型、优化器===================
    # load model
    # 实例化网络，里面包含：特征提取器 + 分类器 + 域判别器（含 GRL）
    net = CNNModel_1D()

    # setup optimizer
    # 优化器： Adam 统一更新所有参数
    opt = optim.Adam(net.parameters(), lr=lr)

    # ===================损失函数:NLLLOSS==================
    manifold_options = {
                        'intraK': 10,
                        'interK': 20,
                        'Regu': 1,
                        'ReguAlpha': 0.5,
                        'ReguBeta': 0.5,
                        'ReducedDim': 10,
                        'KernelType': 'Gaussian',
                        't': 5,
                        'Kernel': 1
                    }
    # 故障分类器的损失、域分类器的损失都是负对数似然（网络最后会 log_softmax）
    loss_cls = torch.nn.CrossEntropyLoss()
    loss_dom = torch.nn.CrossEntropyLoss()

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
        net.train()

        n_iter = min(len(dataloader_source), len(dataloader_target))
        iter_s = iter(dataloader_source)  # 转成迭代器，方便 next()
        iter_t = iter(dataloader_target)

        for i in range(n_iter):
            # 每个 iter 各拿一批源域和目标域

            # **梯度反转系数 α（GRL 核心）
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

            domain_label_s = torch.zeros(bs, dtype=torch.long)

            if cuda:
                s_img = s_img.cuda()
                s_label = s_label.cuda()
                domain_label_s = domain_label_s.cuda()
                    
            # **输入模型 #加入labels算流形学习的graph
            out_cls, out_dom, feature_s = net(s_img, alpha=alpha)
            
            # -------- Manifold regularization Source--------
            # TODO 2：让net返回feature extracter的最后一层特征, 利用特征算流形, 增加一个流形 loss_manifold。
            loss_s_mani = mkglmfa_rkhs_loss(feature_s, s_label, alpha_local=0.5, beta_global=0.5, options=manifold_options)

            # 网络返回：分类 logits，域判别 logits
            err_s_cls = loss_cls(out_cls, s_label)
            err_s_dom = loss_dom(out_dom, domain_label_s)

            # ---- target ----
            # 目标域前向 + 误差【目标域不训练故障分类器，只要域判别输出】
            t_img, t_label = next(iter_t)
            
            bs = t_img.size(0)
            
            # 目标域标签 = 1 
            domain_label = torch.ones(bs, dtype=torch.long)

            if cuda:
                t_img = t_img.cuda()
                t_label = t_label.cuda()
                domain_label = domain_label.cuda()

            _, out_dom, feature_t = net(t_img, alpha=alpha)
            # -------- Manifold regularization target--------
            # TODO 2
            loss_t_mani = mkglmfa_rkhs_loss(feature_t, t_label, alpha_local=0.5, beta_global=0.5, options=manifold_options)
            # loss_t_mani = 0   
            
            err_t_dom = loss_dom(out_dom, domain_label)

            # 原来的误差
            # loss = err_s_cls + err_s_dom + err_t_dom
            # # # 三股误差一起反向，Adam 更新特征提取器、分类器、域判别器全部参数
            # # loss.backward()
            

            lambda_s_cls = 0.9
            lambda_s_dom = 0.6
            lambda_t_dom = 0.6
            lambda_s_mani = 0.5
            lambda_t_mani = 0.5

            loss = (
                    lambda_s_cls * err_s_cls + lambda_s_dom * err_s_dom   +  lambda_t_dom * err_t_dom
                                            +  lambda_s_mani * loss_s_mani + lambda_t_mani * loss_t_mani
                    )
            loss.backward()
            opt.step()

            print(
                f"epoch: {epoch}, [iter: {i + 1} / {n_iter}], "
                f"err_s_label: {err_s_cls.item():.6f}, "
                f"err_s_domain: {err_s_dom.item():.6f}, "
                f"err_t_domain: {err_t_dom.item():.6f}, "
                f"loss_s_mani: {loss_s_mani.item():.6f}, "
                f"loss_t_mani: {loss_t_mani.item():.6f}"
            )

        torch.save(net, model_root / f"zcy_model_epoch_{epoch}.pth")

        #-------一个epoch迭代完，算一次测试准确率---------
        test_new(source_name, epoch)
        test_new(target_name, epoch)

    print("done")


if __name__ == "__main__":
    train()