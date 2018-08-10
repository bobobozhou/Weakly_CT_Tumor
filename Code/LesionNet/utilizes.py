import numpy as np
from skimage.filters import threshold_otsu
from scipy import ndimage
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian
from sklearn.metrics import roc_auc_score, average_precision_score
from skimage.transform import resize
import cv2
import os
import ipdb


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def metric_ROC(output, target):
    """ Calculation of min Ap """
    output_np = output
    target_np = target

    num_class = target.shape[1]
    all_roc_auc = []
    for cid in range(num_class):
        gt_cls = target_np[:, cid].astype('float32')
        pred_cls = output_np[:, cid].astype('float32')

        if all(v == 0 for v in gt_cls):
            roc_auc = float('nan')
        else:
            roc_auc = roc_auc_score(gt_cls, pred_cls, average='weighted')

        all_roc_auc.append(roc_auc)

    all_roc_auc = np.array(all_roc_auc)
    mROC_AUC = all_roc_auc[~np.isnan(all_roc_auc)].mean()
    return [mROC_AUC], [all_roc_auc]


"""Convert Probablity to Binary Segmentation"""
def prob_to_segment(mask_set, hm_set, vol_set, method='Convention'):
    # iteratively convert probablity to binary segmentation
    seg_set = np.zeros((vol_set.shape[0], vol_set.shape[1], vol_set.shape[2], vol_set.shape[3]))

    for i in range(vol_set.shape[0]):
        mask = mask_set[i, :, :, :]
        vol = vol_set[i, :, :, :]
        hm = resize(hm_set[i, :, :, :], 
                    (vol.shape[0], vol.shape[1], vol.shape[2]),
                    preserve_range=True,
                    mode='symmetric')

        '''threshold the predicted segmentation image (a probablity image/volume) using 0.5 hard thresholding'''
        if method is 'Convention':
            if len(np.unique(mask)) is 1:
                thresh = 0.5
            else:
                thresh = threshold_otsu(mask)

            seg = mask > thresh
            seg = np.asarray(seg).astype(np.bool)

        '''3D Dense-connected Conditional Random Field for 3D segmentation using (Probablity volume + CT volume)'''
        if method is '3DCRF':
            seg = mask


        seg_set[i, :, :, :] = seg

    return seg_set



"""Calculate Dice Similarity Coefficient"""
def metric_DSC(output, target):
    """ Calculation of DSC with respect to volume using the slice index (correspond to case) """
    all_DSC = []

    for i in range(output.shape[0]):
        vol_gt = target[i, :, :, :]
        vol_output = output[i, :, :, :]

        dice = DICE(vol_output, vol_gt, empty_score=1.0)
        all_DSC.append(dice)

    all_DSC = np.array(all_DSC)
    mDSC = all_DSC.mean()
    return [mDSC], [all_DSC]


def DICE(im_pred, im_target, empty_score=1.0):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.

    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        Both are empty (sum eq to zero) = empty_score
    """

    # the targert segmentation image
    im_target = np.asarray(im_target).astype(np.bool)

    # calculate the dice using the 1. prediction and 2. ground truth
    if im_pred.shape != im_target.shape:
        raise ValueError("Shape mismatch: im_pred and im_target must have the same shape!")

    im_sum = im_pred.sum() + im_target.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im_pred, im_target)

    return 2. * intersection.sum() / im_sum


"""Volume Display"""
def make_tf_disp_volume(vol_input, vol_target, vol_output, vol_heatmap):
    """
    make the Montage for tensorboard to display 3D volume
    Parameters
    ----------
    output: array-like
        Any array of arbitrary size from the model prediction with case-index.
    targte:  
        Any array of arbitrary size (ground truth segmentation) with case-index
        * Need to be same size as output

    Return
    ----------
    dictionary for inputting the tf logger function, displaying montage
    """

    vol_heatmap = resize(vol_heatmap, (vol_input.shape[0], vol_input.shape[1], vol_input.shape[2]), preserve_range=True, mode='symmetric')

    if vol_input.shape != vol_target.shape or vol_input.shape != vol_output.shape or vol_input.shape != vol_heatmap.shape:
        raise ValueError("Shape mismatch: Image & Prediction & Ground-Truth & Heatmap must have the same shape!")

    montage_input = vol_to_montage(vol_input)
    montage_output = vol_to_montage(vol_output)
    montage_target = vol_to_montage(vol_target)
    montage_heatmap = vol_to_montage(vol_heatmap)

    montage_input = np.repeat(montage_input[np.newaxis, np.newaxis, :, :], 3, axis=1)
    montage_output = np.repeat(montage_output[np.newaxis, np.newaxis, :, :], 3, axis=1)
    montage_target = np.repeat(montage_target[np.newaxis, np.newaxis, :, :], 3, axis=1)
    montage_heatmap = convert_to_heatmap(montage_heatmap, detail=False)[np.newaxis, :, :, :]

    disp_mat = np.concatenate((montage_input, montage_target, montage_output, montage_heatmap), axis=0)

    return disp_mat


def vol_to_montage(vol):
    n_slice, w_slice, h_slice = np.shape(vol)
    nn = int(np.ceil(np.sqrt(n_slice)))
    mm = nn
    M = np.zeros((mm * h_slice, nn * w_slice)) 

    image_id = 0
    for j in range(mm):
        for k in range(nn):
            if image_id >= n_slice: 
                break
            sliceM = j * w_slice 
            sliceN = k * h_slice
            M[sliceN:sliceN + w_slice, sliceM:sliceM + h_slice] = vol[image_id, :, :]
            image_id += 1

    return M


def convert_to_heatmap(img_gray, detail):
    if not detail:
        # 1
        img_gray = 1 / (1 + np.exp(-img_gray))
        im_color = cv2.applyColorMap((img_gray*255).astype(np.uint8), cv2.COLORMAP_JET)
        return im_color[:, :, ::-1].transpose((2,0,1))

    else:
        # 2
        img_gray = (img_gray - img_gray.min())/(img_gray.max() - img_gray.min()) * 255
        r_heatmap = 128 - 128 * np.sin((2*np.pi)/275 * img_gray)
        b_heatmap = 128 + 128 * np.sin((2*np.pi)/275 * img_gray)
        g_heatmap = 128 - 128 * np.cos((2*np.pi)/275 * img_gray)

        heatmaps_show = np.concatenate((r_heatmap[np.newaxis, :, :],
                                        g_heatmap[np.newaxis, :, :],
                                        b_heatmap[np.newaxis, :, :]), axis=0) 
        return heatmaps_show


""" Save Segmentation results during testing"""
def save_seg_montage(vol_input_set, vol_target_set, vol_output_set, vol_heatmap_set, vol_seg_set, epoch):
    """
    make the Montage for saving the slice + segmentation contour results
    Parameters
    ----------
    output: array-like
        Any array of arbitrary size from the model prediction with case-index.
    targte:  
        Any array of arbitrary size (ground truth segmentation) with case-index
        * Need to be same size as output
    """
    # generate folder named epoch to store validation visualization
    newpath = './_RESULTS/epoch' + str(epoch) + '/'
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    # store the montage visualization
    for i in range(vol_input_set.shape[0]):

        vol_input = vol_input_set[i, :, :, :]
        vol_target = vol_target_set[i, :, :, :]
        vol_output = vol_output_set[i, :, :, :]
        vol_heatmap = resize(vol_heatmap_set[i, :, :, :], (vol_input.shape[0], vol_input.shape[1], vol_input.shape[2]), preserve_range=True, mode='symmetric')
        vol_seg = vol_seg_set[i, :, :, :]

        if vol_input.shape != vol_target.shape or vol_input.shape != vol_output.shape or vol_input.shape != vol_heatmap.shape or vol_input.shape != vol_seg.shape:
            raise ValueError("Shape mismatch: Image & Prediction & Ground-Truth & Heatmap & Final Segmentation must have the same shape!")

        plot_data_3d(vol_input.transpose((1,2,0)), savepath=newpath + str(i) + '_volume.png')
        plot_data_3d(vol_target.transpose((1,2,0)), savepath=newpath + str(i) + '_target.png')
        plot_data_3d(vol_output.transpose((1,2,0)), savepath=newpath + str(i) + '_mask.png')
        plot_data_3d(vol_heatmap.transpose((1,2,0)), savepath=newpath + str(i) + '_heatmap.png')
        plot_data_3d(vol_seg.transpose((1,2,0)), savepath=newpath + str(i) + '_segfinal.png')

    return 

def plot_data_3d(vol, savepath):
    """
    Generate an image for 3D data.
    1) show the corresponding 2D slices.
    """

    # Draw
    slides = plot_slides(vol)

    # Save
    cv2.imwrite(savepath, slides)


def plot_slides(vol, _range=None, colored=False):
    """Plot the 2D slides of 3D data"""
    v = vol

    # Rescale the value of voxels into [0, 255], as unsigned byte
    if _range == None:
        v_n = v / (np.max(np.abs(v)) + 0.0000001)
        v_n = (128 + v_n * 127).astype(np.uint8)

    else:
        v_n = (v - _range[0]) / (_range[1] - _range[0])
        v_n = (v_n * 255).astype(np.uint8)

    # Plot the slides
    h, w, d = v.shape
    side_w = int(np.ceil(np.sqrt(d)))
    side_h = int(np.ceil(float(d) / side_w))

    board = np.zeros(((h + 1) * side_h, (w + 1) * side_w, 3))
    for i in range(side_h):
        for j in range(side_w):
            if i * side_w + j >= d:
                break

            img = v_n[:, :, i * side_w + j]
            img = np.repeat(img[:,:,np.newaxis], 3, axis=2)

            board[(h + 1) * i + 1: (h + 1) * (i + 1), (w + 1) * j + 1: (w + 1) * (j + 1), :] = img

    # Return a 2D array representing the image pixels
    return board.astype(int)







# def plot_data_3d(vol_input, vol_output, vol_gt, savepath):
#     """
#     Generate an image for 3D data.
#     1) show the corresponding 2D slices.
#     """

#     # Draw
#     slides = plot_slides(vol_input, vol_output, vol_gt)

#     # Save
#     cv2.imwrite(savepath, slides)


# def plot_slides(vol_input, vol_output, vol_gt, _range=None, colored=False):
#     """Plot the 2D slides of 3D data"""
#     v = vol_input
#     vol_output = vol_output.astype(np.uint8)
#     vol_gt = vol_gt.astype(np.uint8)

#     # Rescale the value of voxels into [0, 255], as unsigned byte
#     if _range == None:
#         v_n = v / (np.max(np.abs(v)) + 0.0000001)
#         v_n = (128 + v_n * 127).astype(np.uint8)

#     else:
#         v_n = (v - _range[0]) / (_range[1] - _range[0])
#         v_n = (v_n * 255).astype(np.uint8)

#     # Plot the slides
#     h, w, d = v.shape
#     side_w = int(np.ceil(np.sqrt(d)))
#     side_h = int(np.ceil(float(d) / side_w))

#     board = np.zeros(((h + 1) * side_h, (w + 1) * side_w, 3))
#     for i in range(side_h):
#         for j in range(side_w):
#             if i * side_w + j >= d:
#                 break

#             img = v_n[:, :, i * side_w + j]
#             img = np.repeat(img[:,:,np.newaxis], 3, axis=2)

#             contours_pred, _ = cv2.findContours(vol_output[:, :, i * side_w + j].copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#             contours_gt, _ = cv2.findContours(vol_gt[:, :, i * side_w + j].copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#             cv2.drawContours(img, contours_pred, -1, (0, 200, 0), 1)
#             cv2.drawContours(img, contours_gt, -1, (0, 0, 200), 1)

#             board[(h + 1) * i + 1: (h + 1) * (i + 1), (w + 1) * j + 1: (w + 1) * (j + 1), :] = img

#     # Return a 2D array representing the image pixels
#     return board.astype(int)
