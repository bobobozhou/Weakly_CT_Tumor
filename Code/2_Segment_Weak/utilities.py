from mayavi import mlab
import torch
from torch.autograd import Variable
from torch.autograd import Function
from torchvision import models
from torchvision import utils
import numpy as np
import cv2
from scipy import ndimage


def plot_data_3d(voxel, savepath):
    """
    Generate an image for 3D data.
    1) show the corresponding 2D slices.
    """

    # Draw
    slides = plot_slides(voxel)

    # Save
    cv2.imwrite(savepath, slides)


def plot_data_cam_3d(voxel, cam, savepath):
    """
    Generate an image for 3D data overlapped with the CAM heatmap.
    1) show the corresponding 2D slices.
    """

    # Resize the CAM
    cam_zoom = ndimage.zoom(cam, zoom=[float(x) / y for x, y in zip(voxel.shape, cam.shape)])

    # Draw the lower half
    slides = plot_slides(cam_zoom, colored=True)

    # Save
    cv2.imwrite(savepath, slides)


def plot_slides(v, _range=None, colored=False):
    """Plot the 2D slides of 3D data"""

    # Rescale the value of voxels into [0, 255], as unsigned byte
    if _range == None:
        v_n = v / (np.max(np.abs(v)) + 0.0000001)
        v_n = (128 + v_n * 127).astype(int)
    else:
        v_n = (v - _range[0]) / (_range[1] - _range[0])
        v_n = (v_n * 255).astype(int)

    # Plot the slides
    h, w, d = v.shape
    side_w = int(np.ceil(np.sqrt(d)))
    side_h = int(np.ceil(float(d) / side_w))

    board = np.zeros(((h + 1) * side_h, (w + 1) * side_w, 3))
    if colored:  # we mix jet colormap for positive part, and use pure grey-scale for negative part
        for i in range(side_h):
            for j in range(side_w):
                if i * side_w + j >= d:
                    break
                values = v_n[:, :, i * side_w + j]
                block1 = cv2.applyColorMap(np.uint8(np.maximum(0, values - 128) * 2), cv2.COLORMAP_JET)
                block2 = np.minimum(128, values)[:, :, np.newaxis] * np.ones((1, 1, 3))
                block = (block1 * np.maximum(0, values - 128)[:, :, np.newaxis] / 128. + block2 * np.minimum(128, 256 - values)[:, :, np.newaxis] / 128.).astype(int)
                board[(h + 1) * i + 1: (h + 1) * (i + 1), (w + 1) * j + 1: (w + 1) * (j + 1), :] = block
    else:  # we just use pure grey-scale for all pixels
        for i in range(side_h):
            for j in range(side_w):
                if i * side_w + j >= d:
                    break
                for k in range(3):
                    board[(h + 1) * i + 1: (h + 1) * (i + 1), (w + 1) * j + 1: (w + 1) * (j + 1), k] = v_n[:, :, i * side_w + j]

    # Return a 2D array representing the image pixels
    return board.astype(int)


def plot_data_cam_2d(image, cam, savepath):
    """Overlap the CAM heatmap on a 2D image"""

    cam_zoom = cv2.resize(cam, (image.shape[1], image.shape[0]))

    # Here for natural images, we only focus on positive values of CAM
    cam_n = np.maximum(cam_zoom, 0) / np.max(cam_zoom)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam_n), cv2.COLORMAP_JET)
    overlapped = np.float32(heatmap) + np.float32(image)
    figure = np.uint8(255 * overlapped / np.max(overlapped))
    cv2.imwrite(savepath, figure)

    return figure


def preprocess_image(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = Variable(preprocessed_img, requires_grad=True)
    return input


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite("examples/respond_cam.jpg", np.uint8(255 * cam))