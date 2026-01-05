import torch.utils.data as data
from PIL import Image
import os
import pickle
import torch


class GetLoader_L(data.Dataset):
    def __init__(self, data_root, data_list, transform=None):
        self.root = data_root
        self.transform = transform


        self.data_path = os.path.join(data_root, data_list)

        with open(self.data_path, "rb") as f:
            self.data_list = pickle.load(f)

        self.n_data = len(self.data_list)

        self.img_paths = []
        self.img_labels = []

        for data in self.data_list:
            self.img_paths.append(data['value']) 
            self.img_labels.append(data['label']) 

        a = 0

    def __getitem__(self, item): # (读取单条数据)
        # 这是 DataLoader 在训练过程中不断调用的核心方法：
        
        # TODO：【要读信号，修改这部分】
        # 索引定位：根据传入的 item（索引）找到对应的图片路径和标签。
        imgs, labels = self.img_paths[item], self.img_labels[item]  # 根据传入的 item（索引）找到对应的图片路径和标签。

        imgs = torch.tensor(imgs, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)

        # 数据转换 (Transform)：如果定义了 transform（如缩放、归一化、转张量等），则对图片进行处理。
        if self.transform is not None:
            imgs = self.transform(imgs)
            labels = int(labels) # 标签转换：将字符型的标签转换为 int 整数类型。

        return imgs, labels  # 返回结果：返回处理后的图片张量和标签。

    def __len__(self):
        # 返回数据集的总样本数，让 PyTorch 知道一轮训练（Epoch）有多少数据。
        return self.n_data
