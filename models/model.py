import torch
import torch.nn as nn
from .functions import ReverseLayerF
# from .functions import ManifoldLayer_BatchLevel
# from MKGLMFA.graph import m_locaglob_torch


class CNNModel(nn.Module):

    def __init__(self):
        super(CNNModel, self).__init__()

        # ===== 特征提取器 feature extractor =====
        self.feature = nn.Sequential() # 空的顺序容器
        #  # 第 1 组卷积：3 通道→64 通道，核 5×5，stride=1，padding=0 → 输出尺寸 24×24
        self.feature.add_module('f_conv1', nn.Conv2d(3, 64, kernel_size=5))  #（in, out, kernel size）
        # 批归一化
        self.feature.add_module('f_bn1', nn.BatchNorm2d(64))
        # 2×2 最大池化 → 12×12
        self.feature.add_module('f_pool1', nn.MaxPool2d(2))
        # ReLU 激活，inplace=True 省显存
        self.feature.add_module('f_relu1', nn.ReLU(True))
        # 还可以用其他激活函数
            # ReLU：简单粗暴、速度快、负半轴“死亡”风险；
            # SiLU：平滑门控、梯度更友好、精度略高，但多一次 sigmoid 计算。
            # 大模型/追求 SOTA → 无脑 SiLU；资源卡得很死的小终端 → ReLU 依旧真香。
        
        # 第 2 组卷积：64→50 通道，核 5×5 → 8×8
        self.feature.add_module('f_conv2', nn.Conv2d(64, 50, kernel_size=5))
        self.feature.add_module('f_bn2', nn.BatchNorm2d(50))
        # 随机 dropout 整张特征图（SpatialDropout）防止过拟合
        self.feature.add_module('f_drop1', nn.Dropout2d())
        self.feature.add_module('f_pool2', nn.MaxPool2d(2))
        self.feature.add_module('f_relu2', nn.ReLU(True))
        # TODO 1:  在这里增加Transformer layer，注意，层的输入和输出的维度
        # TODO 2:  提取完【特征】后，用【特征】计算流形


        # ===== 标签分类器（数字 0-9） =====
        self.class_classifier = nn.Sequential()
        # 把 50×4×4=800 维拉平后接全连接
        self.class_classifier.add_module('c_fc1', nn.Linear(50 * 4 * 4, 100))
        # self.class_classifier.add_module('c_fc1', nn.Linear(10, 100))   # 为了manifold layer新改的
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout2d())
        self.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(100, 10))
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax()) # 输出 log-prob，配合 NLLLoss

        # ===== 域分类器（source vs target） =====
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(50 * 4 * 4, 100))
        # self.domain_classifier.add_module('d_fc1', nn.Linear(10, 100))  # 为了manifold layer新改的
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1)) # 2 类 log-prob
        #TODO 1:用流形学习降维
        # self.manifold_layer = ManifoldLayer_BatchLevel(
        #                                 options={
        #                                     'intraK': 10,
        #                                     'interK': 20,
        #                                     'Regu': 1,
        #                                     'ReguAlpha': 0.5,
        #                                     'ReguBeta': 0.5,
        #                                     'ReducedDim': 10,
        #                                     'KernelType': 'Gaussian',
        #                                     't': 5,
        #                                     'Kernel': 1
        #                                 }
        #                             )


    def forward(self, input_data, labels, alpha):
        B = input_data.size(0)
        # input_data: 假设原始是 1×28×28，但网络第一层需要 3 通道
        input_data = input_data.expand(B, 3, 28, 28)
        # 提取公共特征 50×4×4
        feature = self.feature(input_data) # 8*50*4*4
        # 拉平成 800 维向量
        feature = feature.view(-1, 50 * 4 * 4) # 8*800
        
        #TODO 1:用流形学习降维
        # ---------- 降维 Manifold projection (non-differentiable) ----------
        # manifold_feature = self.manifold_layer(feature, labels)
        # reverse_feature = ReverseLayerF.apply(manifold_feature, alpha)
        # class_output = self.class_classifier(manifold_feature)

        #TODO 2:构建流形学习loss
        # -------- 构图 Local / Global Laplacian （no grad） --------

        # ---------- DANN ----------
        # 梯度反转层：正向传播不变，反向传播时梯度 * -alpha
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        # 数字分类结果
        class_output = self.class_classifier(feature)
        # 域分类结果
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output, feature #, L, Ln
