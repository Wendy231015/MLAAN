import torch.nn as nn
import torch.nn.functional as F


class AuxClassifier(nn.Module):
    # AuxClassifier这个类是自己定义的
    def __init__(self, inplanes,class_num=10, widen=1, feature_dim=128):
        super(AuxClassifier, self).__init__()

        assert inplanes in [16, 32, 64, 512, 768]

        self.feature_dim = feature_dim

        self.criterion = nn.CrossEntropyLoss()
        self.fc_out_channels = class_num

        self.head = nn.Sequential(
                    nn.Conv2d(inplanes,feature_dim,1,1,0,bias=False),
                    nn.BatchNorm2d(feature_dim),
                    nn.ReLU(),
                    nn.Conv2d(feature_dim,feature_dim,1,1,0,bias=False),
                    nn.BatchNorm2d(feature_dim),
                    nn.ReLU(),
                    nn.Conv2d(feature_dim, feature_dim, 1, 1, 0,bias=False),
                    nn.BatchNorm2d(feature_dim),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1,1)),
                    nn.Flatten(),
                    nn.Linear(feature_dim,feature_dim * 4),
                    # nn.Linear(输入，输出）
                    nn.BatchNorm1d(feature_dim * 4),
                    nn.ReLU(inplace=True),
                    nn.Linear(feature_dim * 4,feature_dim * 4),
                    nn.BatchNorm1d(feature_dim * 4),
                    nn.ReLU(inplace=True),
                    nn.Linear(feature_dim * 4, feature_dim * 4),
                    nn.BatchNorm1d(feature_dim * 4),
                    nn.ReLU(inplace=True),
                    nn.Linear(feature_dim * 4,class_num)
        )
        # nn.Sequential来定义一个网络头部（head），头部由一系列卷积层、批归一化层、激活函数和线性层组成。最后一个线性层的输出通道数为class_num，用于进行分类。
        # nn.Sequential是一个用于构建神经网络的容器，nn.Sequential中的模块按照顺序执行，每个模块的输出作为下一个模块的输入
        #  nn.Linear(feature_dim,feature_dim * 4),输出通道是4,该线性层将输入特征映射到一个更高维度的特征空间

    # def forward(self, x, target):
    #     # forward就是代表开始写前向传播的代码了
    #     x = F.adaptive_avg_pool2d(x,(8,8))
    #     # 输入x首先经过一个自适应平均池化层（F.adaptive_avg_pool2d(x,(8,8)）进行特征提取，将输入的特征图大小调整为8x8
    #     features = self.head(x)
    #     # 通过head将特征传回给头部，头部你上面定义过的，进行进一步的特征提取和处理
    #     loss = self.criterion(features, target)
    #     # 使用self.criterion计算特征和目标之间的损失值loss
    #     return loss
    #     # 返回loss



# ViT
    def forward(self, x, target):
        if x.dim() == 2: # ViT的输出通常是2D
            x = x.unsqueeze(-1).unsqueeze(-1) # 添加两个维度，使其变成 [batch_size, channels, 1, 1]
        elif x.dim() != 4:
            raise ValueError("Unsupported input dimension. Expected 2D or 4D input.")

        features = self.head(x) # 经过head模块处理
        loss = self.criterion(features, target) # 计算损失
        return loss