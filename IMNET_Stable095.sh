#!/usr/bin/env bash


# Path to the data where the train/val folders are
PATH_TO_DATA=XXX
python3 main_prune_imagenet.py --config configs/ImageNet/resnet50/SNIP50.json --target_ratio 0.95 --act relu --gpu 0 --scaled_init


# After running the above command there should be a checkpoint popping up in  the same directpry under "runs"
PATH_TO_CHECKP=runs/pruning/ImageNet/resnet50/imagenet_resnet50_SNIP/checkpoint/prune_imagenet_resnet50_r0.95_it0.pth.tar
python3 main_finetune_imagenet.py --data $PATH_TO_DATA --arch resnet50 --resume_pruned $PATH_TO_CHECKP --batch-size 32 --grad_loop 8
