import cv2
import numpy as np
import pydicom
from mask_functions import rle2mask,mask2rle
import tensorflow  as tf



class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, file_path_list, labels, batch_size=32,
                 img_size=256, channels=1, shuffle=True):
        self.file_path_list = file_path_list
        self.labels = labels
        self.batch_size = batch_size
        self.img_size = img_size
        self.channels = channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'denotes the number of batches per epoch'
        return int(np.floor(len(self.file_path_list)) / self.batch_size)

    def __getitem__(self, index):
        'generate one batch of data'
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # get list of IDs
        file_path_list_temp = [self.file_path_list[k] for k in indexes]
        # generate data
        X, y = self.__data_generation(file_path_list_temp)
        # return data
        return X, y

    def on_epoch_end(self):
        'update ended after each epoch'
        self.indexes = np.arange(len(self.file_path_list))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, file_path_list_temp):
        'generate data containing batch_size samples'
        X = np.empty((self.batch_size, self.img_size, self.img_size, self.channels))
        y = np.empty((self.batch_size, self.img_size, self.img_size, self.channels))

        for idx, file_path in enumerate(file_path_list_temp):

            id = file_path.split('/')[-1][:-4]
            rle = self.labels.get(id)
            image = pydicom.read_file(file_path).pixel_array
            image_resized = cv2.resize(image, (self.img_size, self.img_size))
            image_resized = np.array(image_resized, dtype=np.float64)

            X[idx,] = np.expand_dims(image_resized, axis=2)

            # if there is no mask create empty mask
            # notice we are starting of with 1024 because we need to use the rle2mask function
            if rle is None:
                mask = np.zeros((1024, 1024))
            else:
                if len(rle) == 1:
                    mask = rle2mask(rle[0], 1024, 1024).T
                else:
                    mask = np.zeros((1024, 1024))
                    for r in rle:
                        mask = mask + rle2mask(r, 1024, 1024).T

            mask_resized = cv2.resize(mask, (self.img_size, self.img_size))
            y[idx,] = np.expand_dims(mask_resized, axis=2)

        # normalize
        X = X / 255
        y = y / 255

        return X, y