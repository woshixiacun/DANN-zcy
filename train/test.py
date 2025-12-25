import os
import torch.backends.cudnn as cudnn
import torch.utils.data
from torchvision import transforms
from dataset.data_loader import GetLoader
from torchvision import datasets


def test(dataset_name, epoch):
    assert dataset_name in ['MNIST', 'mnist_m']

    model_root = os.path.join('..', 'models')
    image_root = os.path.join('..', 'dataset', dataset_name)

    cuda = True
    cudnn.benchmark = True  #让 cuDNN 自动选最快算法（输入尺寸固定时开）
    batch_size = 128
    image_size = 28
    alpha = 0  # 测试阶段域分类器没用，GRL 系数直接给 0

    """load data"""

    img_transform_source = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))  ## MNIST 官方均值方差
    ]) # 里把 MNIST 也 expand 成 3 通道喂网络，因此实际输入尺寸都是 3×28×28。

    img_transform_target = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))   # # 把 0~1 拉到 [-1,1]
    ])

    # 换成信号数据之后，这一部分要改
    if dataset_name == 'mnist_m':
        test_list = os.path.join(image_root, 'mnist_m_test_labels.txt')

        dataset = GetLoader(   # 自定义 loader，读“图片路径+标签”列表
            data_root=os.path.join(image_root, 'mnist_m_test'),
            data_list=test_list,
            transform=img_transform_target
        )
    else:
        dataset = datasets.MNIST(
            root='../dataset',
            train=False,
            transform=img_transform_source,
        )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8
    )

    """ training """
    # 加载训练好的权重，epoch 用来拼接权重文件名，告诉函数“我要测第几个 epoch 训出来的模型”。
    my_net = torch.load(os.path.join(
        model_root, 'mnist_mnistm_model_epoch_' + str(epoch) + '.pth'
    ))
    my_net = my_net.eval()

    if cuda:
        my_net = my_net.cuda()

    len_dataloader = len(dataloader)
    data_target_iter = iter(dataloader) # 手工迭代器写法（pytorch 0.3 时代常见）

    i = 0
    n_total = 0
    n_correct = 0

    while i < len_dataloader:

        # test model using target data
        data_target = data_target_iter.next()  # 取一个 batch
        t_img, t_label = data_target

        batch_size = len(t_label)

        input_img = torch.FloatTensor(batch_size, 3, image_size, image_size)
        class_label = torch.LongTensor(batch_size)

        if cuda:
            t_img = t_img.cuda()
            t_label = t_label.cuda()
            input_img = input_img.cuda()
            class_label = class_label.cuda()

        input_img.resize_as_(t_img).copy_(t_img)
        class_label.resize_as_(t_label).copy_(t_label)

        # 真正推理：
        class_output, _ = my_net(input_data=input_img, alpha=alpha)
        pred = class_output.data.max(1, keepdim=True)[1]
        n_correct += pred.eq(class_label.data.view_as(pred)).cpu().sum()
        n_total += batch_size

        i += 1

    accu = n_correct.data.numpy() * 1.0 / n_total

    print ('epoch: %d, accuracy of the %s dataset: %f' % (epoch, dataset_name, accu))
