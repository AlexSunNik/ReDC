def iterate(mode, args, loader, model, optimizer, logger, epoch, loss_curve=[], writer=None):
    actual_epoch = epoch - args.start_epoch + args.start_epoch_bias

    block_average_meter = AverageMeter()
    block_average_meter.reset(False)
    average_meter = AverageMeter()
    meters = [block_average_meter, average_meter]

    # switch to appropriate mode
    assert mode in ["train", "val", "eval", "test_prediction", "test_completion"], \
        "unsupported mode: {}".format(mode)
    if mode == 'train':
        model.train()
        lr = helper.adjust_learning_rate(args.lr, optimizer, actual_epoch, args)
    else:
        model.eval()
        lr = 0

    torch.cuda.empty_cache()
    for i, batch_data in enumerate(loader):
        dstart = time.time()
        del batch_data["path"]
        batch_data = {
            key: val.to(device)
            for key, val in batch_data.items() if val is not None
        }

        gt = batch_data[
            'gt'] if mode != 'test_prediction' and mode != 'test_completion' else None
        data_time = time.time() - dstart

        pred = None
        start = None
        gpu_time = 0

        start = time.time()
       
        pred = model((batch_data))
        
        depth_loss, photometric_loss, smooth_loss, mask = 0, 0, 0, None

        if mode == 'train':
            # Loss 1: the direct depth supervision from ground truth label
            # mask=1 indicates that a pixel does not ground truth labels
            depth_loss = depth_criterion(pred, gt)
            loss = depth_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if i % args.print_freq == 0:
                print("loss:", loss.data, " epoch:", epoch, " ", i, "/", len(loader))
            if writer is not None:
                writer.add_scalar('Loss/train', loss.data, epoch*len(loader)+i)
            loss_curve.append(loss.data)
        if mode == "test_completion":
            str_i = str(i)
            path_i = str_i.zfill(10) + '.png'
            path = os.path.join(args.data_folder_save, path_i)
#             vis_utils.save_depth_as_uint16png_upload(pred, path)
            vis_utils.save_depth_as_uint8colored(pred, path)
        if(not args.evaluate):
            gpu_time = time.time() - start
#         # measure accuracy and record loss
        if mode == "val":
            with torch.no_grad():
                mini_batch_size = next(iter(batch_data.values())).size(0)
                result = Result()
                if mode != 'test_prediction' and mode != 'test_completion':
                    result.evaluate(pred.data, gt.data, photometric_loss)
                    [
                        m.update(result, gpu_time, data_time, mini_batch_size)
                        for m in meters
                    ]

                    if mode != 'train':
                        logger.conditional_print(mode, i, epoch, lr, len(loader),
                                         block_average_meter, average_meter)
                    logger.conditional_save_img_comparison(mode, i, batch_data, pred,
                                                       epoch)
                    logger.conditional_save_pred(mode, i, pred, epoch)
    
    if mode == "test_completion":
        print("All images generated")
        
    avg = logger.conditional_save_info(mode, average_meter, epoch)
    is_best = logger.rank_conditional_save_best(mode, avg, epoch)
    if is_best and not (mode == "train"):
        logger.save_img_comparison_as_best(mode, epoch)
    logger.conditional_summarize(mode, avg, is_best)

    return avg, is_best, loss_curve

# ************************************************************************************************************

from PIL import ImageFile
import argparse
import os

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import time

from dataloaders.kitti_loader_original import load_calib, input_options, KittiDepth
from metrics import AverageMeter, Result
import criteria
import helper
import vis_utils
import sys
import cv2
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import numpy as np
# ************************************************************************************************************
parser = argparse.ArgumentParser(description='Sparse-to-Dense')
parser.add_argument('-n',
                    '--network-model',
                    type=str,
                    default="dkn",
                    choices=["e", "pe"],
                    help='choose a model: enet or penet'
                    )
parser.add_argument('--workers',
                    default=8,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs',
                    default=45,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run (default: 100)')
parser.add_argument('--start-epoch',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--start-epoch-bias',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number bias(useful on restarts)')
parser.add_argument('-c',
                    '--criterion',
                    metavar='LOSS',
                    default='l2',
                    choices=criteria.loss_names,
                    help='loss function: | '.join(criteria.loss_names) +
                    ' (default: l2)')
parser.add_argument('-b',
                    '--batch-size',
                    default=4,
                    type=int,
                    help='mini-batch size (default: 1)')
parser.add_argument('--lr',
                    '--learning-rate',
                    default=1e-4,
#                     default=1e-3,
                    type=float,
                    metavar='LR',
                    help='initial learning rate (default 1e-5)')
parser.add_argument('--weight-decay',
                    '--wd',
                    default=1e-5,
                    type=float,
                    metavar='W',
                    help='weight decay (default: 0)')

# parser.add_argument('--weight-decay',
#                     '--wd',
#                     default=5e-4,
#                     type=float,
#                     metavar='W',
#                     help='weight decay (default: 0)')

parser.add_argument('--print-freq',
                    '-p',
                    default=100,
                    type=int,
                    metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('--resume',
                    default='',
                    type=str,
                    metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# parser.add_argument('--data-folder',
#                     default='/data/dataset/kitti_depth/depth',
#                     type=str,
#                     metavar='PATH',
#                     help='data folder (default: none)')

parser.add_argument('--data-folder',
                    default='/shared/rsaas/common/Kitti/',
                    type=str,
                    metavar='PATH',
                    help='data folder (default: none)')

parser.add_argument('--data-folder-rgb',
                    default='/shared/rsaas/common/Kitti/raw_data/',
                    type=str,
                    metavar='PATH',
                    help='data folder rgb (default: none)')

parser.add_argument('--data-folder-save',
                    default='/home/xs15/submit_test/',
                    type=str,
                    metavar='PATH',
                    help='data folder test results(default: none)')
parser.add_argument('-i',
                    '--input',
                    type=str,
                    default='rgbd',
                    choices=input_options,
                    help='input: | '.join(input_options))
parser.add_argument('--val',
                    type=str,
                    default="full",
                    choices=["select", "full"],
                    help='full or select validation set')
parser.add_argument('--jitter',
                    type=float,
                    default=0.1,
                    help='color jitter for images')
parser.add_argument('--rank-metric',
                    type=str,
                    default='rmse',
                    choices=[m for m in dir(Result()) if not m.startswith('_')],
                    help='metrics for which best result is saved')

parser.add_argument('-e', '--evaluate', default='', type=str, metavar='PATH')
parser.add_argument('-f', '--freeze-backbone', action="store_true", default=False,
                    help='freeze parameters in backbone')
parser.add_argument('--test', action="store_true", default=False,
                    help='save result kitti test dataset for submission')
parser.add_argument('--cpu', action="store_true", default=False, help='run on cpu')

#random cropping
parser.add_argument('--not-random-crop', action="store_true", default=False,
                    help='prohibit random cropping')
parser.add_argument('-he', '--random-crop-height', default=320, type=int, metavar='N',
                    help='random crop height')
parser.add_argument('-w', '--random-crop-width', default=1216, type=int, metavar='N',
                    help='random crop height')

#geometric encoding
parser.add_argument('-co', '--convolutional-layer-encoding', default="xyz", type=str,
                    choices=["std", "z", "uv", "xyz"],
                    help='information concatenated in encoder convolutional layers')

#dilated rate of DA-CSPN++
parser.add_argument('-d', '--dilation-rate', default="2", type=int,
                    choices=[1, 2, 4],
                    help='CSPN++ dilation rate')

parser.add_argument('--rbf', action="store_true", default=False,
                    help='RBF interpolation')

parser.add_argument('--nearest', action="store_true", default=False,
                    help='Nearest Grid interpolation')

parser.add_argument('--pe', action="store_true", default=False,
                    help='Nearest Grid interpolation')

args = parser.parse_args()
args.result = os.path.join('..', 'results')
args.use_rgb = ('rgb' in args.input)
args.use_d = 'd' in args.input
args.use_g = 'g' in args.input
args.val_h = 352
args.val_w = 1216
print(args)
kitti_data_folder = '/shared/rsaas/common/Kitti/'
args.data_folder = kitti_data_folder
# define loss functions
if (args.criterion == 'l2'):
    depth_criterion = criteria.MaskedMSELoss()
elif (args.criterion == 'l1'):
    depth_criterion = criteria.MaskedL1Loss()
elif (args.criterion == 'both'):
    depth_criterion = criteria.MaskedBothLoss()
else:
    print("Unrecognized Type")
    exit()
    
checkpoint = None
is_eval = False

# ************************************************************************************************************
test_dataset = KittiDepth('test_completion', args)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=1,
    pin_memory=True)
# ************************************************************************************************************

logger = helper.logger(args)
writer = SummaryWriter()
if checkpoint is not None:
    logger.best_result = checkpoint['best_result']
    del checkpoint
print("=> logger created.")
# from dkn_models import *
# from small_dkn import *
import torch.optim

args.network_model = 'pe'

from redc import ReDC
model = ReDC(args)
# ckpt_path = "/data/pretrained_models/model_best.pth.tar"
ckpt_path = "/home/xs15/LIDAR-DEPTH/results/input=rgbd.criterion=both.lr=0.001.bs=8.wd=1e-05.jitter=0.1.time=2022-07-04@12-07/model_best.pth.tar"
ckpt = torch.load(ckpt_path)
# Load Model Parameters
device = torch.device("cuda")
model.load_state_dict(ckpt['model'])
model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.99))

# optimizer.load_state_dict(ckpt['optimizer'])
# scheduler = lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.2)

# ***********************************************************************************************************
loss_curve = []
start_epoch = args.start_epoch
# start_epoch = ckpt['epoch'] + 1
iterate("test_completion", args, test_loader, model, optimizer, logger, epoch=0, writer=writer)


