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
    source_root = proj_root / "dataset" / source_name
    target_root = proj_root / "dataset" / target_name
    
    model_root = proj_root / "models"

    cuda = True
    cudnn.benchmark = True

    lr = 1e-3
    batch_size = 512
    image_size = 28
    n_epoch = 100

    manual_seed = random.randint(1, 10_000)
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)

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

    ds_source = datasets.MNIST(
        root=str(proj_root / "dataset"),
        train=True,
        transform=img_tf_source,
        download=False,
    )
    dl_source = torch.utils.data.DataLoader(
        ds_source, batch_size=batch_size, shuffle=True, num_workers=8
    )

    train_list = target_root / "mnist_m_train_labels.txt"
    ds_target = GetLoader(
        data_root=str(target_root / "mnist_m_train"),
        data_list=str(train_list),
        transform=img_tf_target,
    )
    dl_target = torch.utils.data.DataLoader(
        ds_target, batch_size=batch_size, shuffle=True, num_workers=8
    )

    net = CNNModel()
    opt = optim.Adam(net.parameters(), lr=lr)
    loss_cls = torch.nn.NLLLoss()
    loss_dom = torch.nn.NLLLoss()

    if cuda:
        net = net.cuda()
        loss_cls = loss_cls.cuda()
        loss_dom = loss_dom.cuda()

    for p in net.parameters():
        p.requires_grad = True

    for epoch in range(n_epoch):
        n_iter = min(len(dl_source), len(dl_target))
        iter_s = iter(dl_source)
        iter_t = iter(dl_target)

        for i in range(n_iter):
            p = float(i + epoch * n_iter) / n_epoch / n_iter
            alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1

            # ---- source ----
            s_img, s_label = next(iter_s)
            net.zero_grad()
            bs = s_label.size(0)

            input_img = torch.empty(bs, 3, image_size, image_size)
            cls_label = torch.empty(bs, dtype=torch.long)
            dom_label = torch.zeros(bs, dtype=torch.long)

            if cuda:
                s_img = s_img.cuda()
                s_label = s_label.cuda()
                input_img = input_img.cuda()
                cls_label = cls_label.cuda()
                dom_label = dom_label.cuda()

            input_img.copy_(s_img.repeat(1, 3, 1, 1))
            cls_label.copy_(s_label)

            out_cls, out_dom = net(input_img, alpha=alpha)
            err_s_cls = loss_cls(out_cls, cls_label)
            err_s_dom = loss_dom(out_dom, dom_label)

            # ---- target ----
            t_img, _ = next(iter_t)
            bs = t_img.size(0)
            input_img = torch.empty(bs, 3, image_size, image_size)
            dom_label = torch.ones(bs, dtype=torch.long)

            if cuda:
                t_img = t_img.cuda()
                input_img = input_img.cuda()
                dom_label = dom_label.cuda()

            input_img.copy_(t_img)

            _, out_dom = net(input_img, alpha=alpha)
            err_t_dom = loss_dom(out_dom, dom_label)

            loss = err_s_cls + err_s_dom + err_t_dom
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