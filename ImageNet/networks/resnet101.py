import copy
import pdb
import time

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

# k2_infopro = {
#     'resnet101': 0,
#     'resnet152': 5,
#     'resnext101_32x8d': 2,
# }

from .configs import Layer
from .configAux_new import Layer_Aux
from .auxnet101 import AuxClassifier


def conv3x3(in_planes, out_planes, stride=1, groups=1):
    " 3x3 convolution with padding "
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

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

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, arch, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, norm_layer=None, net_layer_dict=None,
                 wx=0.001, wy=0.999, momentum=1.,local_module_num = 1):
        super(ResNet, self).__init__()

        assert arch in ['resnet101']

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.momentum = momentum
        self.local_module_num = local_module_num
        self.inplanes = 64
        self.groups = groups
        self.layers = layers
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.criterion_ce = nn.CrossEntropyLoss()

        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, norm_layer=norm_layer)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, norm_layer=norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.wx = wx
        self.wy = wy

        self.bce_loss = nn.BCELoss()
        self.fc_conv = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(2048),
            nn.ReLU())
        self.fc = nn.Linear(2048, num_classes)
        self.wide = [64, 256, 512, 1024, 2048]
        self.aux_config = Layer_Aux[arch][local_module_num]
        self.config = Layer[arch][local_module_num]

        self.Encoder_Net = self._make_Encoder_Aux_Net()
        self.Aug_Net, self.EMA_Net = self._make_Aux_Net()
        self.Update_Net = self._make_Updata_Net()
      

        for i in range(len(self.EMA_Net)):
            for param in self.EMA_Net[i].parameters():
                param.requires_grad = False

        for item in self.config:
            module_index, layer_index = item
            exec('self.net_classifier_' + str(module_index) + '_' + str(layer_index) +
                 '= AuxClassifier(self.wide[module_index])')

            exec('self.aux_classifier_' + str(module_index) + '_' + str(layer_index) +
                 '= AuxClassifier(self.wide[-1])')

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # self.net_layer_dict = net_layer_dict
        # for layer_index in range(self.net_layer_dict['layer_num']):
        #     exec('self.fc' + str(layer_index)
        #          + " = nn.Linear(self.net_layer_dict['feature_num_list'][layer_index], num_classes)")

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        for net in self.Encoder_Net:
            net = net.cuda()

        for net1, net2 in zip(self.Aug_Net, self.EMA_Net):
            net1 = net1.cuda()
            net2 = net2.cuda()

    def _image_restore(self, normalized_image):
        return normalized_image.mul(self.mask_train_std[:normalized_image.size(0)]) \
            + self.mask_train_mean[:normalized_image.size(0)]

    def _make_layer(self, block, planes, blocks, stride=1, norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _make_Encoder_Aux_Net(self):
        Encoder_Net = nn.ModuleList([])

        Encoder_temp = nn.ModuleList([])

        local_block_index = 0

        # Build Encoder_Net
        for blocks in range(len(self.layers)):
            for layers in range(self.layers[blocks]):
                Encoder_temp.append(eval('self.layer' + str(blocks + 1))[layers])
                if blocks + 1 == self.config[local_block_index][0] \
                        and layers == self.config[local_block_index][1]:
                    Encoder_Net.append(nn.Sequential(*Encoder_temp))

                    Encoder_temp = nn.ModuleList([])
                    local_block_index += 1
        return Encoder_Net

    def _make_Aux_Net(self):
        Aux_Net = nn.ModuleList([])
        Aux_temp = nn.ModuleList([])

        for i in range(self.local_module_num - 1):
            for j in range(len(self.aux_config[i])):
                Aux_temp.append(
                    copy.deepcopy(eval("self.layer" + str(self.aux_config[i][j][0]))[self.aux_config[i][j][1]]))
            Aux_Net.append(nn.Sequential(*Aux_temp))
            Aux_temp = nn.ModuleList([])

        EMA_Net = copy.deepcopy(Aux_Net)
        return Aux_Net, EMA_Net

    def _make_Updata_Net(self):
        Update_Net = nn.ModuleList([])
        Update_temp = nn.ModuleList([])

        for i in range(self.local_module_num - 1):
            for j in range(len(self.aux_config[i])):
                Update_temp.append(eval("self.layer" + str(self.aux_config[i][j][0]))[self.aux_config[i][j][1]])
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
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def forward(self, x, initial_image=None, target=None, criterion=None):
        if self.training:
            # local_module_num = 1 means the E2E training
            if self.local_module_num == 1:
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                x = self.maxpool(x)

                for i in range(len(self.Encoder_Net)):
                    x = self.Encoder_Net[i](x)

                x = self.avgpool(x)
                x = x.view(x.size(0), -1)

                logits = self.fc(x)
                loss = self.criterion_ce(logits, target)
                loss.backward()

                return logits, loss

            else:
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                x = self.maxpool(x)

                y = torch.clone(x)

                for i in range(len(self.Encoder_Net) - 1):
                    if i == 0:
                        x = self.Encoder_Net[i](x)
                        x = x.detach()

                    elif i == 1:
                        x = self.Encoder_Net[i](x)
                        x = x.detach()

                    else:
                        x = self.Encoder_Net[i](x)

                        y = self.Encoder_Net[i - 2](y)
                        z = self.Encoder_Net[i - 1](y)
                        z = self.Encoder_Net[i](z)

                        for j in range(len(self.Aug_Net[i])):
                            if j == 0:
                                z = self.Aug_Net[i][j](z) + self.EMA_Net[i][j](z)
                            else:
                                z = self.Aug_Net[i][j](z)

                        local_x, layer_x = self.config[i]
                        loss_x = eval('self.net_classifier_' + str(local_x) + '_' + str(layer_x))(x, target)
                        loss_y = eval('self.aux_classifier_' + str(local_x) + '_' + str(layer_x))(z, target)
                        loss = self.wx * loss_x + self.wy * loss_y
                        loss.backward()

                        x = x.detach()
                        y = y.detach()

                        for paramEncoder, paramEMA in zip(self.Update_Net[i].parameters(),
                                                          self.EMA_Net[i].parameters()):
                            paramEMA.data = paramEMA.data * self.momentum + paramEncoder.data * (1 - self.momentum)
                # last local module
                x = self.Encoder_Net[-1](x)

                y = self.Encoder_Net[-3](y)
                y = self.Encoder_Net[-2](y)
                y = self.Encoder_Net[-1](y)

                x = self.avgpool(x)
                y = self.avgpool(y)

                x = x.view(x.size(0), -1)
                y = y.view(y.size(0), -1)

                logits_x = self.fc(x)
                logits_y = self.fc(y)
                loss_x = self.criterion_ce(logits_x, target)
                loss_y = self.criterion_ce(logits_y, target)
                loss = self.wx * loss_x + self.wy * loss_y
                loss.backward()
                local_index, layer_index = self.config[-1]
                # momentum update parameters of the last AuxNet
                for paramEncoder, paramEMA in zip(self.Update_Net[-1].parameters(), self.EMA_Net[-1].parameters()):
                    paramEMA.data = paramEMA.data * self.momentum + paramEncoder.data * (1 - self.momentum)
            return logits_y, loss


        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            
            output = self.avgpool(x)
            output = output.view(x.size(0), -1)
            output = self.fc(output)
            
            return output



def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], arch='resnet18', **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], arch='resnet34', **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], arch='resnet50', **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))

    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], arch='resnet101', **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], arch='resnet152', **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


def resnext50_32x4d(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], arch='resnext50_32x4d', groups=32, width_per_group=4, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnext50_32x4d']))
    return model


def resnext101_32x8d(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], arch='resnext101_32x8d', groups=32, width_per_group=8, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnext101_32x8d']))
    return model
