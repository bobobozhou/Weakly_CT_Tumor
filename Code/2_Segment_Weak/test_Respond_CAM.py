import torch
from torch.autograd import Variable
from torch.autograd import Function
from torchvision import models
from torchvision import utils
import transforms_3D.transforms_3d as transforms_3d
from torch.utils.data import DataLoader
from data_loader import *
import cv2
import sys
import numpy as np
from model import *
from utilities import *
import argparse

parser = argparse.ArgumentParser(description='Pytorch: 3D CNN for Classification')

# Model structure setting
parser.add_argument('--model', default='spa_resnet',
                    help='model name: (resnet | preresnet | wideresnet | resnext | densenet)')
parser.add_argument('--model_depth', default=101, type=int,
                    help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
parser.add_argument('--pretrain_path', default='./models_set/pretrained/model_best_spa_resnet.pth.tar', type=str,
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

# Respond-CAM setting
parser.add_argument('--use_cuda', action='store_true', default=True,
                    help='Use NVIDIA GPU acceleration')
parser.add_argument('--image_path', type=str, default='./examples/both.png',
                    help='Input image path')

'''Data loader settings'''
parser.add_argument('--workers', default=48, type=int, metavar='N',
                    help='number of data loading worker')
parser.add_argument('--batch_size', default=1, type=int, metavar='N',
                    help='mini-batch size for training (default: 64)')

'''Set up Data Directory'''
parser.add_argument('--vol_data_dir', default='../../Data/nih_data/volume', type=str, metavar='PATH',
                    help='path to volume data')
parser.add_argument('--mask_data_dir', default='../../Data/nih_data/mask', type=str, metavar='PATH',
                    help='path to mask data')
parser.add_argument('--train_list_dir', default='../../Data/nih_data/dir/train_list.txt', type=str, metavar='PATH',
                    help='path to train data list txt file')
parser.add_argument('--test_list_dir', default='../../Data/nih_data/dir/test_list.txt', type=str, metavar='PATH',
                    help='path to test data list txt file')


class FeatureExtractor():
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            if 'fc' in name:
                x = x.view(x.size(0), -1)
            x = module(x)

            if name in self.target_layers[0]:
                for name_sub, module_sub in module._modules.items():
                    if name_sub in self.target_layers[1]:
                        x.register_hook(self.save_gradient)
                        outputs += [x]

        return outputs, x


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, target_layers):
        self.model = model
        self.feature_extractor = FeatureExtractor(self.model, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations, output = self.feature_extractor(x)
        output = output.view(output.size(0), -1)
        return target_activations, output


class RespondCam:
    def __init__(self, model, target_layer_names, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3, 4))[0, :]
        cam = np.ones(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            # cam += w * target[i, :, :, :] * target[i, :, :, :]
            cam += target[i, :, :, :]

        cam = np.maximum(cam, 0)
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam


if __name__ == '__main__':
    global args
    args = parser.parse_args()

    # imread the 3D lesion and RECIST mask using Pytorch's Dataloader
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

    # setup the trained model
    model_res = generate_model(args, PreTrain=True)

    for i, (input_vol, class_vec) in enumerate(val_loader):
        input = Variable(input_vol, requires_grad=True)
        class_vec = class_vec.numpy()

        # setup the Respond-CAM, like target layer
        respond_cam_res = RespondCam(model=model_res, target_layer_names=['layer4', '4'], use_cuda=args.use_cuda)

        # generate the attention 3D mask
        if np.sum(class_vec[0]) == 0:
            target_index = None
        else:
            target_index = np.where(class_vec[0] == 1)[0][0]  # If None, returns the map for the highest scoring category, Otherwise, targets the requested index.

        mask = respond_cam_res(input, target_index)

        # visualiza the mask in a slice-by-slice manner
        V = input.data.numpy()[0,0].transpose([1,2,0])
        M = mask.transpose([1,2,0])
        plot_data_cam_3d(V, M, savepath='./_RESULTS/' + str(i) + 'test_cam.png')
        plot_data_3d(V, savepath='./_RESULTS/' + str(i) + 'test_vol.png')

        a = 1
