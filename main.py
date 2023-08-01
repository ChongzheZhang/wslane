import os
import comet_ml
import cv2
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import argparse
import numpy as np
import random
from wslane.utils.config import Config
from wslane.engine.runner import Runner
from wslane.datasets import build_dataloader


def main():
    if "http_proxy" in os.environ:
        del os.environ["http_proxy"]
    if "https_proxy" in os.environ:
        del os.environ["https_proxy"]
    args = parse_args()
    # os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(gpu) for gpu in args.gpus)

    cfg = Config.fromfile(args.config)
    cfg.gpus = args.gpus

    cfg.load_from = args.load_from
    cfg.resume_from = args.resume_from
    cfg.finetune_from = args.finetune_from
    cfg.view = args.view
    cfg.seed = args.seed
    cfg.no_comet = args.no_comet

    # cfg.cls_loss_weight = args.cls_loss_weight
    # if 'xyt_loss_weight' in cfg:
    #     cfg.xyt_loss_weight = args.xyt_loss_weight
    # cfg.num_lane_loss_weight = args.num_lane_loss_weight
    # if 'tri_loss_weight' in cfg:
    #     cfg.tri_loss_weight = args.tri_loss_weight
    # cfg.iou_loss_weight = args.iou_loss_weight
    # cfg.seg_loss_weight = args.seg_loss_weight
    # if isinstance(cfg.reg_loss_weight, list):
    #     cfg.reg_loss_weight[0] = args.regW0
    #     cfg.reg_loss_weight[1] = args.regW1
    #     cfg.reg_loss_weight[2] = args.regW2
    # else:
    #     cfg.reg_loss_weight = args.reg_loss_weight
    cfg.optimizer.lr = args.lr

    cfg.ema_rate = args.ema_rate

    # cfg.work_dirs = args.work_dirs if args.work_dirs else cfg.work_dirs
    cfg.work_dirs = os.path.join('./outputs', args.work_dirs)

    cudnn.benchmark = True

    runner = Runner(cfg)

    if args.mode == 'train':
        runner.train()
    else:
        runner.test(other_data=args.other_data)

    # if args.validate:
    #     runner.validate()
    # elif args.test:
    #     runner.test()
    # else:
    #     runner.train()


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('mode', choices=['train', 'test'], help="Train or test?")
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work_dirs', type=str, help='output dirs', required=True)
    parser.add_argument('--load_from', default=None, help='the checkpoint file to load from')
    parser.add_argument('--resume_from', default=None, help='the checkpoint file to resume from')
    parser.add_argument('--finetune_from', default=None, help='the checkpoint file to resume from')
    parser.add_argument('--view', action='store_true', help='whether to view')
    parser.add_argument("--other_data", choices=["debug", "porsche"], help="eval other data")
    parser.add_argument('--gpus', type=int, default='1')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--no_comet', action='store_true', help='disable comet')

    # parser.add_argument('--reg_loss_weight', type=float, default=1.0, help='reg_loss_weight')
    # parser.add_argument('--cls_loss_weight', type=float, default=2.0, help='cls_loss_weight')
    # parser.add_argument('--xyt_loss_weight', type=float, default=0.1, help='clrnet xyt_loss_weight')
    # parser.add_argument('--seg_loss_weight', type=float, default=1.0, help='seg_loss_weight')
    # parser.add_argument('--num_lane_loss_weight', type=float, default=1.0, help='num_lane_loss_weight')
    # parser.add_argument('--tri_loss_weight', type=float, default=1.0, help='tri_loss_weight')
    # parser.add_argument('--iou_loss_weight', type=float, default=3.0, help='clrnet iou_loss_weight')
    # parser.add_argument('--regW0', type=float, default=2.0, help='clrnet reg_loss_weight 0')
    # parser.add_argument('--regW1', type=float, default=1.0, help='clrnet reg_loss_weight 1')
    # parser.add_argument('--regW2', type=float, default=0.5, help='clrnet reg_loss_weight 2')
    parser.add_argument('--lr', type=float, default=4.0e-5, help='learning rate')
    parser.add_argument('--ema_rate', type=float, default=0.99, help='ema update rate')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main()
