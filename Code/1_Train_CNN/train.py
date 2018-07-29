"""
Tumor segmentation for free using 3D Classification CNN based on Respond-CAM 
by
Bo Zhou,
Carnegie Mellon University,
Merck Sharp & Dohme (MSD),
bzhou2@cs.cmu.edu
"""

import argparse
import os
import shutil
import time
import sys
import ipdb

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import transforms_3D.transforms_3d as transforms_3d
from torch.utils.data import DataLoader

import numpy as np
import cv2
from scipy.ndimage.interpolation import zoom
from sklearn.metrics import roc_auc_score, average_precision_score
from logger import Logger
from data_loader import *
from model import *
from utilizes import *

'''Set up Training Parameters'''
parser = argparse.ArgumentParser(description='Pytorch: 3D CNN for Classification')

# Model structure setting
parser.add_argument('--model', default='spa_resnet',
                    help='model name: (resnet | preresnet | wideresnet | resnext | densenet)')
parser.add_argument('--model_depth', default=101, type=int,
                    help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
parser.add_argument('--pretrain_path', default='./models_set/pretrained/resnet-101-kinetics.pth', type=str,
                    help='Pretrained model (.pth)')

parser.add_argument('--resnet_shortcut', default='B', type=str,
                    help='Shortcut type of resnet (A | B)')
parser.add_argument('--wide_resnet_k', default=2, type=int,
                    help='Wide resnet k')
parser.add_argument('--resnext_cardinality', default=32, type=int,
                    help='ResNeXt cardinality')

parser.add_argument('--n_classes', default=8, type=int,
                    help='Number of classes output')
parser.add_argument('--sample_size', default=112, type=int,
                    help='Height and width of inputs')
parser.add_argument('--sample_duration', default=16, type=int,
                    help='Distance on z-axis of inputs')

# Model training setting
parser.add_argument('--workers', default=48, type=int, metavar='N',
                    help='number of data loading worker')
parser.add_argument('--epochs', default=1000000, type=int, metavar='N',
                    help='number of epochs for training network')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size', default=3, type=int, metavar='N',
                    help='mini-batch size for training (default: 64)')
parser.add_argument('--lr', default=0.0001, type=float, metavar='LR',
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum for the training optimizer')
parser.add_argument('--wd', default=1e-4, type=float, metavar='W',
                    help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

# Data augmentation setting
parser.add_argument('--initial_scale', default=1.0, type=float,
                    help='Initial scale for multiscale cropping')
parser.add_argument('--n_scales', default=5, type=int,
                    help='Number of scales for multiscale cropping')
parser.add_argument('--scale_step', default=0.84089641525, type=float,
                    help='Scale step for multiscale cropping')
parser.add_argument('--train_crop', default='corner', type=str,
                    help='Spatial cropping method in training. random is uniform. corner is selection from 4 corners and 1 center.  (random | corner | center)')

# Training display setting
parser.add_argument('--pf', default=1, type=int, metavar='N',
                    help='training print frequency (default: 10)')
parser.add_argument('--df', default=2, type=int, metavar='N',
                    help='training display image frequency (default: 10)')
parser.add_argument('--ef', default=1, type=int, metavar='N',
                    help='evaluate print frequency (default: 2)')

'''Set up Data Directory'''
parser.add_argument('--vol_data_dir', default='../../Data/nih_data/volume', type=str, metavar='PATH',
                    help='path to volume data')
parser.add_argument('--mask_data_dir', default='../../Data/nih_data/mask', type=str, metavar='PATH',
                    help='path to mask data')
parser.add_argument('--train_list_dir', default='../../Data/nih_data/dir/train_list.txt', type=str, metavar='PATH',
                    help='path to train data list txt file')
parser.add_argument('--test_list_dir', default='../../Data/nih_data/dir/test_list.txt', type=str, metavar='PATH',
                    help='path to test data list txt file')

best_m = 0


def main():
    global args, best_m
    args = parser.parse_args()

    ''' Initialize and load model (models: Dual Densenet structure) '''
    model = generate_model(args, PreTrain=True)
    print(model)

    model.cuda()
    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True

    ''' Define loss function (criterion) and optimizer '''
    criterion = nn.BCEWithLogitsLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.wd)

    ''' Optional: Resume from a checkpoint '''
    if args.resume:
        if os.path.exists(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_m = checkpoint['best_m']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    '''
    Data loading (CT Tumor Dataset):
    1) Training Data
    2) Validation Data
    '''
    # 1) training data
    train_dataset = CTTumorDataset_FreeSeg(vol_data_dir=args.vol_data_dir,
    									   mask_data_dir=args.mask_data_dir,
                                           list_file=args.train_list_dir,
                                           transform=transforms_3d.Compose(
                                               [transforms_3d.Resize([64, 224, 224]),
                                                transforms_3d.MakeNChannel(3),
                                                transforms_3d.Normalize(mean=[0, 0, 0], std=[2000, 2000, 2000]),
                                                ]))
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers, pin_memory=True)

    # 2) validation data
    val_dataset = CTTumorDataset_FreeSeg(vol_data_dir=args.vol_data_dir,
    	    						     mask_data_dir=args.mask_data_dir,
                                         list_file=args.test_list_dir,
                                         transform=transforms_3d.Compose(
                                             [transforms_3d.Resize([64, 224, 224]),
                                              transforms_3d.MakeNChannel(3),
                                              transforms_3d.Normalize(mean=[0, 0, 0], std=[2000, 2000, 2000]),
                                              ]))
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.workers, pin_memory=True)

    ''' Create logger for recording the training (Tensorboard)'''
    data_logger = Logger('./logs/', name=args.model)

    ''' Training for epochs'''
    TRAIN_STATUS=True
    if TRAIN_STATUS is True:
        for epoch in range(args.start_epoch, args.epochs):
            adjust_learning_rate(optimizer, epoch)

            # train for one epoch
            train(train_loader, model, criterion, optimizer, epoch, data_logger=data_logger)

            # evaluate on validation set
            if epoch % args.ef == 0 or epoch == args.epochs:
                m = validate(val_loader, model, criterion, epoch, data_logger=data_logger)

                # remember best metric and save checkpoint
                is_best = m > best_m
                best_m = max(m, best_m)
                save_checkpoint({
                    'epoch': epoch + 1,
                    'model': args.model,
                    'state_dict': model.state_dict(),
                    'best_m': best_m,
                    'optimizer': optimizer.state_dict(),
                }, is_best, model=args.model)

    ''' Validation'''
    TEST_STATUS = True
    if TEST_STATUS is True:
        epoch = 1
        m = validate(val_loader, model, criterion, epoch, data_logger=data_logger)


def train(train_loader, model, criterion, optimizer, epoch, data_logger=None):
    losses = AverageMeter()

    # switch to training mode and train
    model.train()
    for i, (input_vol,  class_vec) in enumerate(train_loader):
        input_vol_var = torch.autograd.Variable(input_vol, requires_grad=True).cuda()
        class_vec = class_vec.type(torch.FloatTensor).cuda()
        class_vec_var = torch.autograd.Variable(class_vec)

        # 1) output classification
        output = model(input_vol_var)

        # 2) compute the current loss
        loss = criterion(output, class_vec_var)

        # 3) record loss
        losses.update(loss.data[0], input_vol.size(0))

        # 4) compute gradient and do SGD step for optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 5) Record loss (TRAINING)
        # Print the loss, every args.print_frequency during training
        if i % args.pf == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader),
                                                                  loss=losses))

        # Plot the training loss
        data_logger.scalar_summary(tag='train/loss', value=loss, step=i + len(train_loader) * epoch)


def validate(val_loader, model, criterion, epoch, data_logger=None):
    losses = AverageMeter()

    # switch to evaluation mode and evaluate
    model.eval()
    for i, (input_vol, class_vec) in enumerate(val_loader):
        input_vol_var = torch.autograd.Variable(input_vol, requires_grad=True).cuda()
        class_vec = class_vec.type(torch.FloatTensor).cuda()
        class_vec_var = torch.autograd.Variable(class_vec)

        # 1) output classification
        output = model(input_vol_var)

        # 2) compute the current loss
        loss = criterion(output, class_vec_var)

        # 3) record loss
        losses.update(loss.data[0], input_vol.size(0))

        # 5) store all the output, case_ind, gt on validation
        if i == 0:
            cls_pred_all = output.data.cpu().numpy()[:]
            cls_gt_all = class_vec_var.cpu().numpy()[:]

        else:
            cls_pred_all = np.concatenate((cls_pred_all, output.data.cpu().numpy()[:]), axis=0)
            cls_gt_all = np.concatenate((cls_gt_all, class_vec_var.data.cpu().numpy()[:]), axis=0)

    # 5) Calcuate the ROC_AUC for each & the mean ROC_AUC
    mROCAUC, all_ROCAUC = metric_ROC(cls_pred_all, cls_gt_all)

    # 6) Record loss, m; Visualize the segmentation results (VALIDATE)
    # Print the loss, losses_ba, loss_rg, loss_fin, metric_DSC_slice, every args.print_frequency during training
    print('Epoch: [{0}]\t'
          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
          'Metric_ROCAUC {mROCAUC:.3f} ({mROCAUC:.3f})\t'.format(epoch,
                                                                 loss=losses,
                                                                 mROCAUC=mROCAUC[0]))

    # Plot the training loss, loss_ba, loss_rg, loss_fin, metric_DSC_slice
    data_logger.scalar_summary(tag='validate/loss', value=losses.avg, step=epoch)
    data_logger.scalar_summary(tag='validate/ROC_AUC', value=mROCAUC[0], step=epoch)

    return mROCAUC[0]


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.2 ** (epoch // 1000))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, is_best, model=None):
    """Save checkpoint and the current best model"""
    save_path = './models/' + str(model)
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    filename_ckpt = save_path + '/checkpoint_' + str(model) + '.pth.tar'
    filename_best = save_path + '/model_best_' + str(model) + '.pth.tar'

    torch.save(state, filename_ckpt)
    if is_best:
        shutil.copyfile(filename_ckpt, filename_best)


if __name__ == '__main__':
    main()
