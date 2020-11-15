import argparse
import os
import torch
import torch.nn as nn

from models.model_base import ModelBase
from tensorboardX import SummaryWriter
#from models.base.init_utils import weights_init
from utils.common_utils import (get_logger, makedirs, process_config, str_to_list)
from pruner.GraSP_ImageNet import GraSP
from pruner.SNIP_ImageNet import SNIP
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data
from models.base.init_utils import weights_init_EOC, weights_init_kaiming_xavier, weights_init_xavier, \
	weights_init_kaiming_relu, weights_init_kaiming_tanh, weights_init_ord, ord_weights, ord_bias
from utils.network_utils import get_network

def init_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--run', type=str, default='')

    parser.add_argument('--init', type=str, default='kaiming_xavier')
    parser.add_argument('--target_ratio', type=float, default=None)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed_tiny', type=int, default=0)
    parser.add_argument('--scaled_init', action='store_true')
    parser.add_argument('--bn', action='store_true')
    parser.add_argument('--act', type=str, default='relu')
    parser.add_argument('--sigma_w2', type=float, default=None, help='This is only for ordered init')
    args = parser.parse_args()
    runs = None
    if len(args.run) > 0:
        runs = args.run
    config = process_config(args.config, runs)

    #os.environ['CUDA_VISIBLE_DEVICES'] = "{}".format(args.gpu)

    return config, args



def init_logger(config):
    makedirs(config.summary_dir)
    makedirs(config.checkpoint_dir)

    # set logger
    path = os.path.dirname(os.path.abspath(__file__))
    path_model = os.path.join(path, 'models/base/%s.py' % 'vgg')
    path_main = os.path.join(path, 'main_prune_imagenet.py')
    path_pruner = os.path.join(path, 'pruner/%s.py' % config.pruner_file)
    logger = get_logger('log', logpath=config.summary_dir+'/',
                        filepath=path_model, package_files=[path_main, path_pruner])
    logger.info(dict(config))
    writer = SummaryWriter(config.summary_dir)
    return logger, writer


def print_mask_information(mb, logger):
    ratios = mb.get_ratio_at_each_layer()
    logger.info('** Mask information of %s. Overall Remaining: %.2f%%' % (mb.get_name(), ratios['ratio']))
    count = 0
    for k, v in ratios.items():
        if k == 'ratio':
            continue
        logger.info('  (%d) %s: Remaining: %.2f%%' % (count, k, v))
        count += 1


def get_exception_layers(net, exception):
    exc = []
    idx = 0
    for m in net.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            if idx in exception:
                exc.append(m)
            idx += 1
    return tuple(exc)


def main(config, args):
    # init logger
    classes = {
        'cifar10': 10,
        'cifar100': 100,
        'mnist': 10,
        'tiny_imagenet': 200,
        'imagenet': 1000
    }
    logger, writer = init_logger(config)

    # build model
    # model = models.__dict__[config.network]()
    model = get_network(config.network, config.depth, config.dataset, use_bn=config.get('use_bn', args.bn),
                        scaled=args.scaled_init, act=args.act)
    mb = ModelBase(config.network, config.depth, config.dataset, model)
    mb.cuda()

    # preprocessing
    # ====================================== fetch configs ======================================
    ckpt_path = config.checkpoint_dir
    num_iterations = config.iterations

    if args.target_ratio == None:
        target_ratio = config.target_ratio
    else:
        target_ratio = args.target_ratio

    normalize = config.normalize
    # ====================================== fetch exception ======================================
    exception = get_exception_layers(mb.model, str_to_list(config.exception, ',', int))
    logger.info('Exception: ')

    for idx, m in enumerate(exception):
        logger.info('  (%d) %s' % (idx, m))

    # ====================================== fetch training schemes ======================================
    ratio = 1-(1-target_ratio) ** (1.0 / num_iterations)
    learning_rates = str_to_list(config.learning_rate, ',', float)
    weight_decays = str_to_list(config.weight_decay, ',', float)
    training_epochs = str_to_list(config.epoch, ',', int)
    logger.info('Normalize: %s, Total iteration: %d, Target ratio: %.2f, Iter ratio %.4f.' %
                (normalize, num_iterations, target_ratio, ratio))
    logger.info('Basic Settings: ')
    for idx in range(len(learning_rates)):
        logger.info('  %d: LR: %.5f, WD: %.5f, Epochs: %d' % (idx,
                                                              learning_rates[idx],
                                                              weight_decays[idx],
                                                              training_epochs[idx]))


    # ====================================== get dataloader ======================================
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        config.traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=16, pin_memory=False, sampler=None)

    # ====================================== start pruning ======================================

    for iteration in range(num_iterations):
        logger.info('** Target ratio: %.4f, iter ratio: %.4f, iteration: %d/%d.' % (target_ratio,
                                                                                    ratio,
                                                                                    iteration,
                                                                                    num_iterations))

        assert num_iterations == 1
        print("=> Applying weight initialization.")
        mb.model.apply(weights_init_kaiming_xavier)

        print("=> Applying weight initialization(%s)." % config.get('init_method', 'kaiming'))
        print("Iteration of: %d/%d" % (iteration, num_iterations))

        if config.pruner == 'SNIP':
            print('=> Using SNIP')
            masks, scaled_masks = SNIP(mb.model, ratio, trainloader, 'cuda',
                                       num_classes=classes[config.dataset],
                                       samples_per_class=config.samples_per_class,
                                       num_iters=config.get('num_iters', 1),
                                       scaled_init=False)
        elif config.pruner == 'GraSP':
            print('=> Using GraSP')
            masks = GraSP(mb.model, ratio, trainloader, 'cuda',
                          num_classes=classes[config.dataset],
                          samples_per_class=config.samples_per_class,
                          num_iters=config.get('num_iters', 1))

        # ========== register mask ==================
        mb.masks = masks
        # ========== save pruned network ============
        logger.info('Saving..')
        state = {
            'net': mb.model,
            'acc': -1,
            'epoch': -1,
            'args': config,
            'mask': mb.masks,
            'ratio': mb.get_ratio_at_each_layer()
        }
        path = os.path.join(ckpt_path, 'prune_%s_%s%s_r%s_it%d.pth.tar' % (config.dataset,
                                                                           config.network,
                                                                           config.depth,
                                                                           target_ratio,
                                                                           iteration))
        torch.save(state, path)

        # ========== print pruning details ============
        logger.info('**[%d] Mask and training setting: ' % iteration)
        print_mask_information(mb, logger)


if __name__ == '__main__':
    # config = init_config()
    # main(config)
    torch.manual_seed(12)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    config, args = init_config()
    torch.manual_seed(args.seed_tiny)
    main(config, args)
