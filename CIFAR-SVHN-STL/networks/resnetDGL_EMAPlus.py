import copy
import pdb
import time

import torch
import torch.nn as nn
import math

from .configAuxPlus_new import InfoPro_Aux, InfoPro_balanced_memory_Aux
from .config2 import InfoPro , InfoPro_balanced_memory
from .auxDGL import AuxClassifier


def conv3x3(in_planes, out_planes, stride=1):
    " 3x3 convolution with padding "
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion=1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dropout_rate=0):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    # ResNet50第一个卷积层的卷积核大小为1x1，第二个卷积层的卷积核大小为3x3，第三个卷积层的卷积核大小为1x1
    # 这三个卷积层之间都有Batch Normalization和ReLU激活函数
    # Bottleneck的输入和输出通道数相同，因此需要在第一个卷积层和第三个卷积层中使用1x1的卷积核来改变通道数
    # expansion = 4表示Bottleneck中第二个卷积层的输出通道数是输入通道数的4倍

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(inplanes, planes*4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.downsample is not None:
            residual = self.downsample(x)

        x += residual
        x = self.relu(x)

        return x


class InfoProResNet(nn.Module):

    def __init__(self, block, layers, arch, local_module_num, batch_size, image_size=32,
                 balanced_memory=False, dataset='cifar10', class_num=10,
                 wide_list=(16, 16, 32, 64), dropout_rate=0,
                 aux_net_config='1c2f', local_loss_mode='contrast',
                 aux_net_widen=1, aux_net_feature_dim=128,momentum = 0.999):
        # block参数指定了使用的残差块类型，layers参数指定了每个stage中包含的残差块数量，arch参数指定了ResNet的类型（如ResNet-18、ResNet-50等）
        # balanced_memory参数指定了是否使用平衡内存的方式训练模型，dataset参数指定了使用的数据集，class_num参数指定了数据集中的类别数，wide_list参数指定了每个stage中的通道数
        super(InfoProResNet, self).__init__()

        assert arch in ['resnet32', 'resnet110'], "This repo supports resnet32 and resnet110 currently. " \
                                                  "For other networks, please set network configs in .configs."
        self.widelist = wide_list
        self.inplanes = wide_list[0]
        self.dropout_rate = dropout_rate
        self.feature_num = wide_list[-1]
        self.class_num = class_num
        self.local_module_num = local_module_num
        self.layers = layers
        self.momentum = momentum

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, wide_list[1], layers[0])
        self.layer2 = self._make_layer(block, wide_list[2], layers[1], stride=2)
        self.layer3 = self._make_layer(block, wide_list[3], layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.feature_num, self.class_num)
        self.fc64 = nn.Linear(self.widelist[3], self.class_num)
        self.Flatten = nn.Flatten()
        self.criterion_ce = nn.CrossEntropyLoss()


        try:
            self.infopro_config = InfoPro_balanced_memory[arch][dataset][local_module_num] \
                if balanced_memory else InfoPro[arch][local_module_num]
        except:
            raise NotImplementedError

        try:
            self.aux_config = InfoPro_balanced_memory_Aux[arch][dataset][local_module_num] \
                if balanced_memory else InfoPro_Aux[arch][local_module_num]
        except:
            raise NotImplementedError

        for j in range(55):
            for module_index in range(1,4):
                for layer_index in range(len(self.layer1)):

                    exec('self.aux_classifier_' + str(module_index) + '_' + str(layer_index) + '_' + str(j) +
                         '= AuxClassifier(wide_list[module_index], class_num=class_num, '
                         'widen=aux_net_widen, feature_dim=aux_net_feature_dim)')
                # 通过循环创建了多个辅助分类器（Auxiliary Classifier），每个辅助分类器都有自己的命名规则，如“self.aux_classifier_1_0”表示第一个模块的第一个层的辅助分类器。

        # 这里遍历EMA时不计算梯度
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # 计算卷积核的数量
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # 让卷积层的权重服从正态分布
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                # 归一层初始化权重为1
                m.bias.data.zero_()
                # 归一层初始化偏置项为0
        # 这段在初始化卷积层和归一层的权重

        if 'cifar' in dataset:
            self.mask_train_mean = torch.Tensor([x / 255.0 for x in [125.3, 123.0, 113.9]]).view(1, 3, 1, 1).expand(
                batch_size, 3, image_size, image_size
            ).cuda()
            self.mask_train_std = torch.Tensor([x / 255.0 for x in [63.0, 62.1, 66.7]]).view(1, 3, 1, 1).expand(
                batch_size, 3, image_size, image_size
            ).cuda()
        else:
            self.mask_train_mean = torch.Tensor([x / 255.0 for x in [127.5, 127.5, 127.5]]).view(1, 3, 1, 1).expand(
                batch_size, 3, image_size, image_size
            ).cuda()
            self.mask_train_std = torch.Tensor([x / 255.0 for x in [127.5, 127.5, 127.5]]).view(1, 3, 1, 1).expand(
                batch_size, 3, image_size, image_size
            ).cuda()
        self.Encoder_Net = self._make_Encoder_Aux_Net()
        self.Aug_Net,self.EMA_Net = self._make_Aux_Net()
        self.Update_Net = self._make_Update_Net()
        for i in range(len(self.EMA_Net)):
            for param in self.EMA_Net[i].parameters():
                param.requires_grad = False

        for net in self.Encoder_Net:
            net = net.cuda()

        for net1, net2 in zip(self.Aug_Net, self.EMA_Net):
            net1 = net1.cuda()
            net2 = net2.cuda()

    def _image_restore(self, normalized_image):
        return normalized_image.mul(self.mask_train_std[:normalized_image.size(0)]) \
               + self.mask_train_mean[:normalized_image.size(0)]
    # 原始图像恢复，标准差+均值？

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dropout_rate=self.dropout_rate))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dropout_rate=self.dropout_rate))

        return nn.Sequential(*layers)

    def _make_Encoder_Aux_Net(self):
        Encoder_Net = nn.ModuleList([])

        Encoder_temp = nn.ModuleList([])

        local_block_index = 0

        # Build Encoder_Net
        for blocks in range(len(self.layers)):
            for layers in range(self.layers[blocks]):
                Encoder_temp.append(eval('self.layer' + str(blocks + 1))[layers])
                if blocks + 1 == self.infopro_config[local_block_index][0] \
                        and layers == self.infopro_config[local_block_index][1]:
                    Encoder_Net.append(nn.Sequential(*Encoder_temp))

                    Encoder_temp = nn.ModuleList([])
                    local_block_index += 1
        return Encoder_Net
        # 通过遍历layers列表中的每个元素，将每个元素中的卷积层和池化层添加到Encoder_temp中
        # 当遍历到infopro_config中指定的位置时，将Encoder_temp中的层打包成一个Sequential对象添加到Encoder_Net中
        # 最终返回Encoder_Net作为编码器网络

    def _make_Aux_Net(self):
        Aux_Net = nn.ModuleList([])

        Aux_temp = nn.ModuleList([])

        for i in range(self.local_module_num - 1):
            for j in range(len(self.aux_config[i])):
                Aux_temp.append(copy.deepcopy(eval('self.layer' + str(self.aux_config[i][j][0]))[self.aux_config[i][j][1]]))
            Aux_Net.append(nn.Sequential(*Aux_temp))
            Aux_temp = nn.ModuleList([])
        EMA_Net = copy.deepcopy(Aux_Net)
        return Aux_Net,EMA_Net

    def _make_Update_Net(self):
        Update_Net = nn.ModuleList([])

        Update_temp = nn.ModuleList([])

        for i in range(self.local_module_num - 1):
            for j in range(len(self.aux_config[i])):
                Update_temp.append(eval('self.layer' + str(self.aux_config[i][j][0]))[self.aux_config[i][j][1]])
            Update_Net.append(nn.Sequential(*Update_temp))
            Update_temp = nn.ModuleList([])
        return Update_Net


    def forward_original(self, img):
        x = self.conv1(img)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc64(x)
    # 通过全连接层输出一个64维的向量

    def forward(self, img, target=None,ixx_1=0, ixy_1=0,
                ixx_2=0, ixy_2=0):
        if self.training:
            # local_module_num = 1 means the E2E training
            if self.local_module_num == 1:
                x = self.conv1(img)
                x = self.bn1(x)
                x = self.relu(x)

                for i in range(len(self.Encoder_Net)):
                    x = self.Encoder_Net[i](x)

                x = self.avgpool(x)
                x = x.view(x.size(0),-1)

                logits = self.fc64(x)
                # 这段代码将经过平均池化和形状改变后的张量x传递给全连接层self.fc64，得到输出logits。
                loss = self.criterion_ce(logits,target)
                # 损失函数用的交叉熵损失函数
                loss.backward()

                return logits,loss

            else:
                x = self.conv1(img)
                x = self.bn1(x)
                x = self.relu(x)
                for i in range(len(self.Encoder_Net) - 1):
                    x = self.Encoder_Net[i](x)
                    for j in range(len(self.Aug_Net[i])):
                        if j == 0:
                            y = self.EMA_Net[i][j](x) + self.Aug_Net[i][j](x)
                        else:
                            y = self.Aug_Net[i][j](y)
                    loss = eval('self.aux_classifier_' + str(3) + '_' + str(len(self.layer1) - 1) + '_' + str(i))(y,target)
                    loss.backward()
                    x = x.detach()
                    # momentum update parameters of AuxNet
                    for paramEncoder, paramEMA in zip(self.Update_Net[i].parameters(),self.EMA_Net[i].parameters()):
                        paramEMA.data = paramEMA.data * self.momentum + paramEncoder.data * (1 - self.momentum)
                # last local module
                x = self.Encoder_Net[-1](x)
                x = self.avgpool(x)
                x = x.view(x.size(0), -1)

                logits = self.fc64(x)
                loss = self.criterion_ce(logits, target)
                loss.backward()
                local_index, layer_index = self.infopro_config[-1]
                # momentum update parameters of the last AuxNet
                for paramEncoder, paramEMA in zip(self.Update_Net[-1].parameters(), self.EMA_Net[-1].parameters()):
                    paramEMA.data = paramEMA.data * self.momentum + paramEncoder.data * (1 - self.momentum)
            return logits, loss

        else:
            x = self.conv1(img)
            x = self.bn1(x)
            x = self.relu(x)

            for i in range(len(self.Encoder_Net)):
                x = self.Encoder_Net[i](x)

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            logits = self.fc64(x)
            loss = self.criterion_ce(logits, target)
            return logits, loss


def resnet20(**kwargs):
    model = InfoProResNet(BasicBlock, [3, 3, 3], arch='resnet20', **kwargs)
    return model
# 定义了一个名为resnet20的函数，该函数返回一个ResNet-20模型
# 该模型使用了InfoProResNet类，该类使用了BasicBlock作为残差块，并且在ResNet的第二个参数中传递了[3, 3, 3]，这表示该模型有3个残差块，每个残差块中有3个卷积层

def resnet32(**kwargs):
    model = InfoProResNet(BasicBlock, [5, 5, 5], arch='resnet32', **kwargs)
    return model


def resnet44(**kwargs):
    model = InfoProResNet(BasicBlock, [7, 7, 7], arch='resnet44', **kwargs)
    return model


def resnet56(**kwargs):
    model = InfoProResNet(BasicBlock, [9, 9, 9], arch='resnet56', **kwargs)
    return model


def resnet110(**kwargs):
    model = InfoProResNet(BasicBlock, [18, 18, 18], arch='resnet110', **kwargs)
    return model


def resnet1202(**kwargs):
    model = InfoProResNet(BasicBlock, [200, 200, 200], arch='resnet1202', **kwargs)
    return model


def resnet164(**kwargs):
    model = InfoProResNet(Bottleneck, [18, 18, 18], arch='resnet164', **kwargs)
    return model


def resnet1001(**kwargs):
    model = InfoProResNet(Bottleneck, [111, 111, 111], arch='resnet1001', **kwargs)
    return model

# if __name__ == "__main__":
#     net = resnet110(local_module_num=54, batch_size=256, image_size=32,
#                    balanced_memory=False, dataset='cifar10', class_num=10,
#                    wide_list=(16, 16, 32, 64), dropout_rate=0,
#                    aux_net_config='1c2f', local_loss_mode='contrast',
#                    aux_net_widen=1, aux_net_feature_dim=128)
#     net = net.cuda()
#     x = torch.ones(4,3,32,32).cuda()
#     target = torch.zeros(4).long().cuda()
#     print(net(x, target))