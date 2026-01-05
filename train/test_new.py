import os
import torch.backends.cudnn as cudnn
import torch.utils.data
from torchvision import transforms
from dataset.data_loader_adopt import GetLoader_L
from torchvision import datasets
from pathlib import Path

def test_new(dataset_name: str, epoch: int) -> None:
    assert dataset_name in {"zcy", "mnist_m"}

    root = Path(__file__).resolve().parent.parent  # 项目根目录

    model_root = root / "models"
    image_root = root / "dataset" / dataset_name

    cuda = True
    cudnn.benchmark = True
    batch_size = 8
    image_size = 28
    alpha = 0.0  # 测试阶段 GRL 系数给 0

    # ---------- 数据 ----------
    if dataset_name == "zcy":
        dataset = GetLoader_L(
            data_root=image_root,
            data_list="test.pkl",
            transform=None,
        )
    else:
        pass
        # dataset = datasets.MNIST(
        #     root=str(root / "dataset"),
        #     train=False,
        #     transform=tf_source,
        # )

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=8
    )

    # ---------- 模型 ----------
    # 加载训练好的权重，epoch 用来拼接权重文件名，告诉函数“我要测第几个 epoch 训出来的模型”。
    net = torch.load(model_root / f"zcy_model_epoch_{epoch}.pth")
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

            out, _ ,_ = net(img ,alpha=alpha)
            pred = out.argmax(dim=1)
            n_correct += pred.eq(label).sum().item()
            n_total += label.size(0)

    acc = n_correct / n_total
    print(f"epoch: {epoch}, accuracy of the {dataset_name} dataset: {acc:.6f}")