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
from skimage.transform import resize
from sklearn.metrics import roc_auc_score, average_precision_score
from logger import Logger
from data_loader import *
from model import *
from utilizes import *
from myloss import *

'''Set up Training Parameters'''
parser = argparse.ArgumentParser(description='Pytorch: 3D CNN for Classification')

# Model structure setting
parser.add_argument('--model', default='lesionnet',
                    help='model name')
parser.add_argument('--pretrain_path', default='./pretrained/resnet-101-kinetics.pth', type=str,
                    help='Pretrained model (.pth)')
parser.add_argument('--n_classes', default=8, type=int,
                    help='Number of classes output')
parser.add_argument('--workers', default=48, type=int, metavar='N',
                    help='number of data loading worker')
parser.add_argument('--epochs', default=1000000, type=int, metavar='N',
                    help='number of epochs for training network')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size', default=6, type=int, metavar='N',
                    help='mini-batch size for training (default: 64)')
parser.add_argument('--lr', default=0.0001, type=float, metavar='LR',
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum for the training optimizer')
parser.add_argument('--wd', default=1e-4, type=float, metavar='W',
                    help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

# Training display setting
parser.add_argument('--pf', default=1, type=int, metavar='N',
                    help='training print frequency (default: 10)')
parser.add_argument('--df', default=50, type=int, metavar='N',
                    help='training display image frequency (default: 10)')
parser.add_argument('--ef', default=4, type=int, metavar='N',
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

dict_cls = {0:'None', 1:'bone', 2:'abdomen', 3:'mediastunum', 4:'liver', 5:'lung', 6:'kidney', 7:'soft-tissue', 8:'pelvis'}

best_m = 0


def main():
    global args, best_m
    args = parser.parse_args()

    ''' Initialize and load model '''
    model = lesion_net()
    print(model)

    model.cuda()
    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True

    ''' Define loss function (criterion) and optimizer '''
    criterion_pred = nn.BCEWithLogitsLoss().cuda()
    criterion_mask = SoftDiceLoss().cuda()
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
                                           transform_vol=transforms_3d.Compose(
                                               [transforms_3d.Resize([32, 64, 64], order=1),
                                                transforms_3d.MakeNChannel(1),
                                                transforms_3d.Normalize(mean=[0, 0, 0], std=[3000, 3000, 3000]),
                                                ]),
                                           transform_mask=transforms_3d.Compose(
                                               [transforms_3d.Resize([32, 64, 64], order=0),
                                                transforms_3d.MakeNChannel(1),
                                                ])
                                           )
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers, pin_memory=True)

    # 2) validation data
    val_dataset = CTTumorDataset_FreeSeg(vol_data_dir=args.vol_data_dir,
                                         mask_data_dir=args.mask_data_dir,
                                         list_file=args.test_list_dir,
                                         transform_vol=transforms_3d.Compose(
                                             [transforms_3d.Resize([32, 64, 64], order=1),
                                              transforms_3d.MakeNChannel(1),
                                              transforms_3d.Normalize(mean=[0, 0, 0], std=[3000, 3000, 3000]),
                                              ]),
                                         transform_mask=transforms_3d.Compose(
                                             [transforms_3d.Resize([32, 64, 64], order=0),
                                              transforms_3d.MakeNChannel(1),
                                              ])
                                         )
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
            train(train_loader, model, criterion_mask, criterion_pred, optimizer, epoch, data_logger=data_logger)

            # evaluate on validation set
            if epoch % args.ef == 0 or epoch == args.epochs:
                m_dsc, m_roc = validate(val_loader, model, criterion_mask, criterion_pred, epoch, data_logger=data_logger)
                m = m_dsc

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
        m_dsc, m_roc = validate(val_loader, model, criterion_mask, criterion_pred, epoch, data_logger=data_logger)


def train(train_loader, model, criterion_mask, criterion_pred, optimizer, epoch, data_logger=None):
    losses_seg = AverageMeter()
    losses_cls = AverageMeter()
    losses = AverageMeter()

    # switch to training mode and train
    model.train()
    for i, (input_vol, mask, class_vec) in enumerate(train_loader):
        input_vol_var = torch.autograd.Variable(input_vol, requires_grad=True).cuda()
        mask_var = torch.autograd.Variable(mask, requires_grad=True).cuda()
        class_vec = class_vec.type(torch.FloatTensor).cuda()
        class_vec_var = torch.autograd.Variable(class_vec)

        # 1) output classification
        output_mask0, output_mask1, output_mask2, output_hm = model(input_vol_var)

        x = output_hm.view([output_hm.shape[0], output_hm.shape[1],
                            output_hm.shape[2] * output_hm.shape[3] * output_hm.shape[4]])   # training stage -> topk selection
        output_topk, _ = torch.topk(x, 120, -1)
        kk = np.random.randint(low=1, high=120, size=1)
        output_pred = output_topk[:,:,kk].squeeze()

        # 2) compute the current loss
        loss_mask0 = criterion_mask(output_mask0, mask_var)
        loss_mask1 = criterion_mask(output_mask1, mask_var)
        loss_mask2 = criterion_mask(output_mask2, mask_var)
        loss_mask = loss_mask0 + loss_mask1 + loss_mask2

        loss_pred = criterion_pred(output_pred, class_vec_var)
        loss = 10 * loss_mask  + 1 * loss_pred

        # 3) record loss
        losses_seg.update(loss_mask.data[0], input_vol.size(0))
        losses_cls.update(loss_pred.data[0], input_vol.size(0))
        losses.update(loss.data[0], input_vol.size(0))

        # 4) compute gradient and do SGD step for optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 5) Record loss: segmentation & classification (TRAINING)
        # Print the loss, every args.print_frequency during training
        if i % args.pf == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Loss_Seg {loss_seg.val:.4f} ({loss_seg.avg:.4f})\t'
                  'Loss_Cls {loss_cls.val:.4f} ({loss_cls.avg:.4f})\t'.format(epoch, i, len(train_loader), 
                                                                              loss=losses,
                                                                              loss_seg=losses_seg,
                                                                              loss_cls=losses_cls))

        # Plot the training loss
        data_logger.scalar_summary(tag='train/loss', value=loss, step=i + len(train_loader) * epoch)
        data_logger.scalar_summary(tag='train/loss_mask', value=loss_mask, step=i + len(train_loader) * epoch)
        data_logger.scalar_summary(tag='train/loss_pred', value=loss_pred, step=i + len(train_loader) * epoch)

        # 6) Display the Visualization: Segmentation & Heatmap with classification
        if i % args.df == 0:
            # only display the first one in the batch
            cls_ind = 0
            if sum(class_vec.data.cpu().numpy()[0]) != 0:
                cls_ind = np.where(class_vec.data.cpu().numpy()[0]==1)[0][0] + 1

                disp = make_tf_disp_volume(vol_input=input_vol_var.data.cpu().numpy()[0, 0, :, :, :],
                                           vol_target=mask_var.data.cpu().numpy()[0, 0, :, :, :], 
                                           vol_output=output_mask0.data.cpu().numpy()[0, 0, :, :, :],
                                           vol_heatmap=output_hm.data.cpu().numpy()[0, cls_ind-1, :, :, :])          # only display the class's heatmap
                
                tag_inf = '_epoch:' + str(epoch) + ' _iter:' + str(i)
                data_logger.image_summary(tag='train/' + tag_inf + '-cls:' + str(cls_ind) + dict_cls[cls_ind], images=disp, step=i + len(train_loader) * epoch)


def validate(val_loader, model, criterion_mask, criterion_pred, epoch, data_logger=None):
    losses_seg = AverageMeter()
    losses_cls = AverageMeter()
    losses = AverageMeter()

    # switch to evaluation mode and evaluate
    model.eval()
    for i, (input_vol, mask, class_vec) in enumerate(val_loader):
        input_vol_var = torch.autograd.Variable(input_vol, requires_grad=True).cuda()
        mask_var = torch.autograd.Variable(mask, requires_grad=True).cuda()
        class_vec = class_vec.type(torch.FloatTensor).cuda()
        class_vec_var = torch.autograd.Variable(class_vec)

        # 1) output classification
        output_mask0, output_mask1, output_mask2, output_hm = model(input_vol_var)

        find_max = nn.MaxPool3d(kernel_size=output_hm.size()[2:])   # testing stage -> maximum selection
        output_pred = find_max(output_hm)
        output_pred = output_pred.view(output_pred.shape[0], output_pred.shape[1])

        # 2) compute the current loss
        loss_mask0 = criterion_mask(output_mask0, mask_var)
        loss_mask1 = criterion_mask(output_mask1, mask_var)
        loss_mask2 = criterion_mask(output_mask2, mask_var)
        loss_mask = loss_mask0 + loss_mask1 + loss_mask2

        loss_pred = criterion_pred(output_pred, class_vec_var)
        loss = 2 * loss_mask  + 0 * loss_pred

        # 3) record loss
        losses_seg.update(loss_mask.data[0], input_vol.size(0))
        losses_cls.update(loss_pred.data[0], input_vol.size(0))
        losses.update(loss.data[0], input_vol.size(0))

        # 5) store all the output, case_ind, gt on validation
        if i == 0:
            input_vol_all = input_vol_var.data.cpu().numpy()[:, 0, :, :, :]
            mask_all = mask_var.data.cpu().numpy()[:, 0, :, :, :]
            output_mask_all = output_mask0.data.cpu().numpy()[:, 0, :, :, :]
            output_hm_all = output_hm.data.cpu().numpy()[:, 0, :, :, :]

            class_vec_all = class_vec_var.data.cpu().numpy()[:]
            output_pred_all = output_pred.data.cpu().numpy()[:]

        else:
            input_vol_all = np.concatenate((input_vol_all, input_vol_var.data.cpu().numpy()[:, 0, :, :, :]), axis=0)
            mask_all = np.concatenate((mask_all, mask_var.data.cpu().numpy()[:, 0, :, :, :]), axis=0)
            output_mask_all = np.concatenate((output_mask_all, output_mask0.data.cpu().numpy()[:, 0, :, :, :]), axis=0)
            output_hm_all = np.concatenate((output_hm_all, output_hm.data.cpu().numpy()[:, 0, :, :, :]), axis=0)

            class_vec_all = np.concatenate((class_vec_all, class_vec_var.data.cpu().numpy()[:]), axis=0)
            output_pred_all = np.concatenate((output_pred_all, output_pred.data.cpu().numpy()[:]), axis=0)

    # 5) 1.Convert Vol+Mask+HM to Segmentation ; 2.Calculate mDSC ; 3. Calculate mROC_AUC
    seg_all = prob_to_segment(mask_set=output_mask_all, hm_set=output_hm_all, vol_set=input_vol_all, method='Convention')

    mDSC, all_DSC = metric_DSC(seg_all, mask_all)
    mROCAUC, all_ROCAUC = metric_ROC(output_pred_all, class_vec_all)

    # 6) Record loss, m; Visualize the segmentation results (VALIDATE)
    # Print the loss, losses_ba, loss_rg, loss_fin, metric_DSC_slice, every args.print_frequency during training
    print('Epoch: [{0}]\t'
          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
          'Loss_Seg {loss_seg.val:.4f} ({loss_seg.avg:.4f})\t'
          'Loss_Cls {loss_cls.val:.4f} ({loss_cls.avg:.4f})\t'
          'Metric_DSC {mDSC:.3f} ({mDSC:.3f})\t'
          'Metric_ROCAUC {mROCAUC:.3f} ({mROCAUC:.3f})\t'.format(epoch,
                                                                 loss=losses,
                                                                 loss_seg=losses_seg,
                                                                 loss_cls=losses_cls,
                                                                 mROCAUC=mROCAUC[0],
                                                                 mDSC=mDSC[0]))

    # Plot the training loss, loss_ba, loss_rg, loss_fin, metric_DSC_slice
    data_logger.scalar_summary(tag='validate/loss', value=losses.avg, step=epoch)
    data_logger.scalar_summary(tag='validate/DSC', value=mDSC[0], step=epoch)
    data_logger.scalar_summary(tag='validate/ROC_AUC', value=mROCAUC[0], step=epoch)

    # 7) Save the visualization results in the _RESULTS folder
    save_seg_montage(vol_input_set=input_vol_all, vol_target_set=mask_all, 
                     vol_output_set=output_mask_all, vol_heatmap_set=output_hm_all,
                     vol_seg_set=seg_all, epoch=epoch)

    return mDSC[0], mROCAUC[0]


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
