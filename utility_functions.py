import pandas as pd
import numpy as np
import zipfile
import os
import glob
import random
import sys


import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import backend as K
from tqdm import tqdm
import pydicom
import cv2
import matplotlib.patches as mpatches

# mask functions
def mask2rle(img, width, height):
    rle = []
    lastColor = 0;
    currentPixel = 0;
    runStart = -1;
    runLength = 0;

    for x in range(width):
        for y in range(height):
            currentColor = img[x][y]
            if currentColor != lastColor:
                if currentColor == 255:
                    runStart = currentPixel;
                    runLength = 1;
                else:
                    rle.append(str(runStart));
                    rle.append(str(runLength));
                    runStart = -1;
                    runLength = 0;
                    currentPixel = 0;
            elif runStart > -1:
                runLength += 1
            lastColor = currentColor;
            currentPixel+=1;

    return " ".join(rle)
# mask functions
def rle2mask(rle, width, height):
    mask= np.zeros(width* height)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position:current_position+lengths[index]] = 255
        current_position += lengths[index]

    return mask.reshape(width, height)


def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


def dicom_to_dict(dicom_data, file_path, rles_df, encoded_pixels=True):
    data = {}

    # Parse fields with meaningful information
    data['patient_name'] = dicom_data.PatientName
    data['patient_id'] = dicom_data.PatientID
    data['patient_age'] = int(dicom_data.PatientAge)
    data['patient_sex'] = dicom_data.PatientSex
    data['pixel_spacing'] = dicom_data.PixelSpacing
    data['file_path'] = file_path
    data['id'] = dicom_data.SOPInstanceUID

    # look for annotation if enabled (train set)
    if encoded_pixels:
        encoded_pixels_list = rles_df[rles_df['ImageId'] == dicom_data.SOPInstanceUID]['EncodedPixels'].values

        pneumothorax = False
        for encoded_pixels in encoded_pixels_list:
            if encoded_pixels != ' -1':
                pneumothorax = True

        # get meaningful information (for train set)
        data['encoded_pixels_list'] = encoded_pixels_list
        data['has_pneumothorax'] = pneumothorax
        data['encoded_pixels_count'] = len(encoded_pixels_list)

    return data

def boundingbox(img):
    # return max and min of a mask to draw bounding box
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax


def plot_with_mask_and_bbox(file_path, mask_encoded_list, figsize=(20, 10)):
    ds = pydicom.dcmread(file_path)
    pixel_array = ds.pixel_array

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    clahe_pixel_array = clahe.apply(pixel_array)

    # use the masking function to decode RLE
    mask_decoded_list = [rle2mask(mask_encoded, 1024, 1024).T for mask_encoded in mask_encoded_list]

    fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(20, 10))

    # print out the xray
    ax[0].imshow(pixel_array, cmap=plt.cm.bone)
    # print the bounding box
    for mask_decoded in mask_decoded_list:
        # print out the annotated area
        ax[0].imshow(mask_decoded, alpha=0.3, cmap="Reds")
        rmin, rmax, cmin, cmax = boundingbox(mask_decoded)
        bbox = mpatches.Rectangle((cmin, rmin), cmax - cmin, rmax - rmin, linewidth=1, edgecolor='r', facecolor='none')
        ax[0].add_patch(bbox)
    ax[0].set_title('With Mask')

    # plot image with clahe processing with just bounding box and no mask
    ax[1].imshow(clahe_pixel_array, cmap=plt.cm.bone)
    for mask_decoded in mask_decoded_list:
        rmin, rmax, cmin, cmax = boundingbox(mask_decoded)
        bbox = mpatches.Rectangle((cmin, rmin), cmax - cmin, rmax - rmin, linewidth=1, edgecolor='r', facecolor='none')
        ax[1].add_patch(bbox)
    ax[1].set_title('Without Mask - Clahe')

    # plot plain xray with just bounding box and no mask
    ax[2].imshow(pixel_array, cmap=plt.cm.bone)
    for mask_decoded in mask_decoded_list:
        rmin, rmax, cmin, cmax = boundingbox(mask_decoded)
        bbox = mpatches.Rectangle((cmin, rmin), cmax - cmin, rmax - rmin, linewidth=1, edgecolor='r', facecolor='none')
        ax[2].add_patch(bbox)
    ax[2].set_title('Without Mask')
    plt.show()

    def get_test_tensor(file_path, batch_size, img_size, channels):

        X = np.empty((batch_size, img_size, img_size, channels))

        # Store sample
        pixel_array = pydicom.read_file(file_path).pixel_array
        image_resized = cv2.resize(pixel_array, (img_size, img_size))
        image_resized = np.array(image_resized, dtype=np.float64)
        image_resized -= image_resized.mean()
        image_resized /= image_resized.std()
        X[0,] = np.expand_dims(image_resized, axis=2)

        return X


def plot_train(img, mask, pred):
    fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(15, 5))

    ax[0].imshow(img, cmap=plt.cm.bone)
    ax[0].set_title('Chest X-Ray')

    ax[1].imshow(mask, cmap=plt.cm.bone)
    ax[1].set_title('Mask')

    ax[2].imshow(pred, cmap=plt.cm.bone)
    ax[2].set_title('Pred Mask')

    plt.show()