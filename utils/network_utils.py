from models.base import (VGG,
                         resnet)


def get_network(network, depth, dataset, use_bn=True, scaled=False, act='relu'):
    if network == 'vgg':
        print('Use batch norm is: %s' % use_bn)
        return VGG(depth=depth, dataset=dataset, batchnorm=use_bn, act=act)
    elif network == 'resnet':
        return resnet(depth=depth, dataset=dataset, scaled=scaled, act=act)
    else:
        raise NotImplementedError('Network unsupported ' + network)


def stablize_bn(net, trainloader, device='cuda'):
    """Iterate over the dataset for stabilizing the
    BatchNorm statistics.
    """
    net = net.train()
    for batch, (inputs, _) in enumerate(trainloader):
        inputs = inputs.to(device)
        net(inputs)
