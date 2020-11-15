import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from utils.common_utils import try_cuda
# from .init_utils import weights_init

__all__ = ['resnet']  # , 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']
_AFFINE = True
#_AFFINE = False


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, scaled=False, act='relu', scaling_fac=15):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, affine=_AFFINE)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, affine=_AFFINE)
        self.scaled = scaled
        self.downsample = None
        self.bn3 = None
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False))
            self.bn3 = nn.BatchNorm2d(self.expansion * planes, affine=_AFFINE)

        if act == 'relu':
            self.activation = F.relu
        elif act == 'elu':
            self.activation = F.elu
        elif act == 'tanh':
            self.activation = F.tanh

        self.scaling_fac = scaling_fac
        if self.scaled:
            print('The scaling factor is {}'.format(self.scaling_fac))


    def forward(self, x):
        # x: batch_size * in_c * h * w
        residual = x
        out = self.activation(self.bn1(self.conv1(x)))
        if self.scaled:
            # TODO change to make it more modular 15 is only for resnet32
            out = self.bn2(self.conv2(out))/math.sqrt(self.scaling_fac)
        else:
            out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.bn3(self.downsample(x))
        out += residual
        out = self.activation(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, scaled=False, act='relu'):
        super(ResNet, self).__init__()
        _outputs = [32, 64, 128]
        scaling_factor = sum(num_blocks)

        self.in_planes = _outputs[0]
        self.act = act
        self.conv1 = nn.Conv2d(3, _outputs[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(_outputs[0], affine=_AFFINE)
        self.layer1 = self._make_layer(block, _outputs[0], num_blocks[0], stride=1, scaled=scaled, act=act, scaling_fac=scaling_factor)
        self.layer2 = self._make_layer(block, _outputs[1], num_blocks[1], stride=2, scaled=scaled, act=act, scaling_fac=scaling_factor)
        self.layer3 = self._make_layer(block, _outputs[2], num_blocks[2], stride=2, scaled=scaled, act=act, scaling_fac=scaling_factor)
        self.linear = nn.Linear(_outputs[2], num_classes)
        self.scaled = scaled
        if act == 'relu':
            print('We are using RELU')
            self.activation = F.relu
        elif act == 'elu':
            print('We are using ElU')
            self.activation = F.elu
        elif act == 'tanh':
            print('We are using TANH')
            self.activation = F.tanh

        #self.apply(weights_init)

    def _make_layer(self, block, planes, num_blocks, stride, scaled, act, scaling_fac):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, scaled, act, scaling_fac))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.activation(self.bn(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet(depth=32, dataset='cifar10', scaled=False, act='relu'):
    if scaled:
        print('#' * 40)
        print('We are scaling the blocks!!!!')
        print('#' * 40)
    assert (depth - 2) % 6 == 0, 'Depth must be = 6n + 2, got %d' % depth
    n = (depth - 2) // 6
    if dataset == 'cifar10':
        num_classes = 10
    elif dataset == 'cifar100':
        num_classes = 100
    elif dataset == 'tiny_imagenet':
        num_classes = 200
    elif dataset == 'imagenet':
        num_classes = 1000
    else:
        raise NotImplementedError('Dataset [%s] is not supported.' % dataset)
    return ResNet(BasicBlock, [n]*3, num_classes, scaled, act)



def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()
