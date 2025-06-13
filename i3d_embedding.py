import torch
import torch.nn as nn
import torchvision
from torchvision.models.video import r3d_18


class I3DEmbedding(nn.Module):
    def __init__(self, output_dim=1024, pretrained=True): # ImageNet Kinetics 预训练权重
        super(I3DEmbedding, self).__init__()
        self.embedding_dim = output_dim

        # ✅ 加载预训练的 I3D backbone（这里用的是简化版 r3d_18）
        self.i3d = r3d_18(pretrained=pretrained) # 接收的向量形式是 [C,T,3,H,W] C-clip, T- time, H - height, W - width

        # ✅ 去掉 avgpool 和分类层，保留时间维度
        self.i3d.avgpool = nn.Identity() # 池化操作会丢失时间特征 我们需要提取时间特征 - 给后续Transformer提供特征值
        self.i3d.fc = nn.Identity() # 不需要概率输出 我们的工作是特征提取

        # ✅ 添加一个线性投影层将通道数映射到 output_dim（如 1024）
        self.linear_proj = nn.Linear(512, output_dim) # I3d这个模型本身是输出维度512 通过转化成1024来满足 Transformer的要求

    def forward(self, x):
        """
        :param x: 输入为视频帧序列 [B, 3, T, H, W]
        :return: 时序特征 [T', output_dim]，或 [B, T', output_dim]（如果保留 batch）
        """
        print(f"🟢 输入 shape: {x.shape}")
        # ✅ 提取中间特征
        features = self.i3d.stem(x) # 第一个卷积层+BN+ReLU+MaxPool，对视频帧进行初步的空间/时间下采样。
        features = self.i3d.layer1(features) # 每一层都是多个 3D 残差块，使用 Conv3D 处理 [T, H, W] 三维结构。每个卷积块会进一步压缩空间分辨率，同时聚合时间上的信息。
        features = self.i3d.layer2(features)
        features = self.i3d.layer3(features)
        features = self.i3d.layer4(features)  # [B, 512, T', 1, 1]

        # ✅ 去除空间维度 [B, 512, T'] 我们不在乎当前画面上的手语的具体位置 只在乎具体内容
        features = torch.mean(features, dim=[3, 4])  # [B, 512, T'] 这里的B是batch_size 一般来说是1 一次处理一个视频文件

        #features = features.squeeze(-1).squeeze(-1)

        if features.ndim == 3:
            features = features.permute(0, 2, 1)  # [B, T', 512] 交换位置
        elif features.ndim == 2:
            features = features.unsqueeze(1)     # 单帧输入 -> [B, 1, 512]
        else:
            raise ValueError(f"❌ Unexpected features shape: {features.shape}")

        # ✅ 映射到指定维度
        features = self.linear_proj(features)  # [B, T', output_dim]
        print(f"🟢 提取后 shape: {features.shape}")

        # ✅ 如果是单个样本（B=1），则 squeeze 掉 batch 维度
        if features.shape[0] == 1:
            return features.squeeze(0)  # [T', output_dim]
        return features  # 多个样本: [B, T', output_dim]


"""
import torch
import torch.nn as nn
import torchvision  # ✅ 这行是你缺失的

from torchvision.models.video import r3d_18

class I3DEmbedding(nn.Module):
    def __init__(self, output_dim=1024, pretrained=True):
        super(I3DEmbedding, self).__init__()
        self.embedding_dim = output_dim  # ✅ 添加这一行！！

        # 加载 I3D backbone
        self.i3d = torchvision.models.video.r3d_18(pretrained=pretrained)
        self.i3d.fc = nn.Identity()  # 移除最后分类层

        # 添加 projection 层将 I3D 输出投影到指定维度
        self.linear_proj = nn.Linear(512, output_dim)  # 注意：此处输入 512 取决于 I3D 输出特征

    def forward(self, x, mask=None):
        features = self.i3d(x)              # (B, 512)
        print(f"[DEBUG] I3D 输出特征维度: {features.shape}")
        return self.linear_proj(features)   # (B, output_dim)

"""