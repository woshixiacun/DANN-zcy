import os
import torch.backends.cudnn as cudnn
import torch.utils.data
from torchvision import transforms
from dataset.data_loader import GetLoader
from torchvision import datasets
from pathlib import Path

def test(dataset_name: str, epoch: int) -> None:
    assert dataset_name in {"MNIST", "mnist_m"}

    root = Path(__file__).resolve().parent.parent  # 项目根目录

    model_root = root / "models"
    image_root = root / "dataset" / dataset_name

    cuda = True
    cudnn.benchmark = True
    batch_size = 128
    image_size = 28
    alpha = 0.0  # 测试阶段 GRL 系数给 0

    # ---------- 数据 ----------
    tf_source = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,)), ### MNIST 官方均值方差
        ]
    )
    tf_target = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )

    if dataset_name == "mnist_m":
        test_list = image_root / "mnist_m_test_labels.txt"
        dataset = GetLoader(
            data_root=str(image_root / "mnist_m_test"),
            data_list=str(test_list),
            transform=tf_target,
        )
    else:
        dataset = datasets.MNIST(
            root=str(root / "dataset"),
            train=False,
            transform=tf_source,
        )

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=8
    )

    # ---------- 模型 ----------
    # 加载训练好的权重，epoch 用来拼接权重文件名，告诉函数“我要测第几个 epoch 训出来的模型”。
    net = torch.load(model_root / f"mnist_mnistm_model_epoch_{epoch}.pth")
    net.eval()
    if cuda:
        net = net.cuda()

    # ---------- 测试 ----------
    n_total = 0
    n_correct = 0

    with torch.no_grad():  # 3.8 推荐显式关闭梯度
        for img, label in loader:
            if cuda:
                img, label = img.cuda(), label.cuda()

            # MNIST 扩到 3 通道
            img = img.repeat(1, 3, 1, 1) if img.size(1) == 1 else img

            out, _ = net(img, alpha=alpha)
            pred = out.argmax(dim=1)
            n_correct += pred.eq(label).sum().item()
            n_total += label.size(0)

    acc = n_correct / n_total
    print(f"epoch: {epoch}, accuracy of the {dataset_name} dataset: {acc:.6f}")