import torch.utils.data as data
from PIL import Image
import os


class GetLoader(data.Dataset):
    def __init__(self, data_root, data_list, transform=None):
        self.root = data_root
        self.transform = transform

        f = open(data_list, 'r')  # 读取文件：打开 data_list（通常是一个 .txt 文件），读取每一行内容。
        data_list = f.readlines()
        f.close()

        self.n_data = len(data_list)

        self.img_paths = []
        self.img_labels = []

        for data in data_list:
            self.img_paths.append(data[:-3]) # 截取每一行的前部分作为图片路径（这里假设路径后有固定长度的后缀或空格）。
            self.img_labels.append(data[-2]) # 取倒数第二个字符作为标签。

    def __getitem__(self, item): # (读取单条数据)
        # 这是 DataLoader 在训练过程中不断调用的核心方法：
        
        # TODO：【要读信号，修改这部分】
        # 索引定位：根据传入的 item（索引）找到对应的图片路径和标签。
        img_paths, labels = self.img_paths[item], self.img_labels[item]  # 根据传入的 item（索引）找到对应的图片路径和标签。
        # 加载图片：使用 PIL.Image.open 打开图片，并强制转换为 RGB 模式（防止灰度图或带透明通道的图导致维度不匹配）。
        imgs = Image.open(os.path.join(self.root, img_paths)).convert('RGB')

        # 数据转换 (Transform)：如果定义了 transform（如缩放、归一化、转张量等），则对图片进行处理。
        if self.transform is not None:
            imgs = self.transform(imgs)
            labels = int(labels) # 标签转换：将字符型的标签转换为 int 整数类型。

        return imgs, labels  # 返回结果：返回处理后的图片张量和标签。

    def __len__(self):
        # 返回数据集的总样本数，让 PyTorch 知道一轮训练（Epoch）有多少数据。
        return self.n_data
