import torch
import torch.nn as nn
from .functions import ReverseLayerF
# from .functions import ManifoldLayer_BatchLevel
# from MKGLMFA.graph import m_locaglob_torch


class CNNModel_1D(nn.Module):

    def __init__(self):
        super(CNNModel_1D, self).__init__()

        # ===== 特征提取器 featureure extractor =====
        self.featureure = nn.Sequential(
            # [B, 1, 1024] → [B, 64, 512]
            nn.Conv1d(1, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.MaxPool1d(2),

            # [B, 64, 512] → [B, 128, 256]
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.MaxPool1d(2),

            # [B, 128, 256] → [B, 256, 128]
            nn.Conv1d(128, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.MaxPool1d(2),
        )
        # TODO 1:  在这里增加Transformer layer，注意，层的输入和输出的维度
        # TODO 2:  提取完【特征】后，用【特征】计算流形

        # ✅ 自动算 featureure_dim（强烈推荐）
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 1024)
            feature = self.featureure(dummy)
            self.featureure_dim = feature.view(1, -1).size(1)


        # ===== 标签分类器 =====
        self.class_classifier = nn.Sequential(
            nn.Linear(self.featureure_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(0.5),

            nn.Linear(256, 10)   # 10 类
            # ⚠️ 不要加 LogSoftmax
        )
        # ===== 域分类器（source vs target） =====
        self.domain_classifier = nn.Sequential(
            nn.Linear(self.featureure_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),

            nn.Linear(256, 2)
        )


    def forward(self, input_data, alpha=1.0):
        # x: [B, 1024]
        x = input_data.unsqueeze(1)   # [B, 1, 1024]
        # 提取公共特征
        feature = self.featureure(x)
        # 拉平
        feature = feature.view(feature.size(0), -1)

        # ---------- DANN ----------
        # 数字分类结果
        class_out = self.class_classifier(feature)
        # 梯度反转层：正向传播不变，反向传播时梯度 * -alpha
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        # 域分类结果
        domain_out = self.domain_classifier(reverse_feature)

        return class_out, domain_out, feature

