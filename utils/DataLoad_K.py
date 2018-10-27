import numpy as np
import csv
import cv2
from PIL import Image
import tensorflow.keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight, shuffle
from sklearn.model_selection import train_test_split
from imgaug import augmenters as iaa
import pandas as pd
import os
from tensorflow.keras.utils import to_categorical

class DataSet:

    def __init__(self, batch_size=16):
        self.train_path = "../data/perprocess/train/"
        self.data = pd.read_csv("../data/train.csv")
        self.batch_size = batch_size
        self.train_dataset_info = []
        self.data_aug = True
        self.SIZE = 299

        for name, labels in zip(self.data['Id'], self.data['Target'].str.split(' ')):
            self.train_dataset_info.append({
                'path': os.path.join(self.train_path, name),
                'labels': np.array([int(label) for label in labels])})
        self.train_dataset_info = np.array(self.train_dataset_info)

        #self.train_laobel = [i['labels'] for i in self.train_dataset_info]

        indexes = np.arange(self.train_dataset_info.shape[0])
        self.train_data_index, self.val_data_index = train_test_split(indexes, test_size=0.10, random_state=8)
        self.train_data = self.train_dataset_info[self.train_data_index]
        self.val_data = self.train_dataset_info[self.val_data_index]

    def augment(self, image):
        augment_img = iaa.Sequential([
            iaa.OneOf([
                iaa.Affine(rotate=0),
                iaa.Affine(rotate=90),
                iaa.Affine(rotate=-90),
                iaa.Fliplr(0.5),
            ])], random_order=True)

        image_aug = augment_img.augment_image(image)
        return image_aug

    def data_load(self, path, shape):
        image = np.array(Image.open(path + '.png'))
        image = cv2.resize(image, (shape[0], shape[1]))
        return image

    def submit_generator(self):
        test_data = pd.read_csv('../data/sample_submission.csv')

        while True:
            for start in range(0, len(test_data), self.batch_size):
                end = min(start + self.batch_size, len(test_data))
                batch_imgs = []
                X_test_batch = test_data[start:end]
                for i in range(len(X_test_batch)):
                    image = self.data_load(X_test_batch[i]['path'], shape=(self.SIZE, self.SIZE, 3))
                    batch_imgs.append(image / 255.)


                yield np.array(batch_imgs, np.float32), X_test_batch

    def train_generator(self, data_aug=True):
        train_data = shuffle(self.train_data)
        while True:
            for start in range(0, len(train_data), self.batch_size):
                end = min(start + self.batch_size, len(train_data))
                batch_imgs = []
                X_train_batch = train_data[start:end]
                batch_labels = np.zeros((len(X_train_batch), 28))
                #batch_labels = []
                for i in range(len(X_train_batch)):
                    image = self.data_load(X_train_batch[i]['path'], shape=(self.SIZE, self.SIZE, 3))
                    if data_aug is True:
                        image = self.augment(image)
                    batch_imgs.append(image/255.)
                    #batch_labels.append(to_categorical(X_train_batch[i]['labels'], 28))
                    batch_labels[i][X_train_batch[i]['labels']] = 1
                    #print(batch_labels)

                yield np.array(batch_imgs, np.float32), batch_labels

    def val_generator(self, data_aug = False):
        train_data = shuffle(self.val_data)
        while True:
            for start in range(0, len(train_data), self.batch_size):
                end = min(start + self.batch_size, len(train_data))
                batch_imgs = []
                #batch_labels = []
                X_train_batch = train_data[start:end]
                batch_labels = np.zeros((len(X_train_batch), 28))
                for i in range(len(X_train_batch)):
                    image = self.data_load(X_train_batch[i]['path'], shape=(self.SIZE, self.SIZE, 3))
                    if data_aug is True:
                        image = self.augment(image)
                    batch_imgs.append(image / 255.)
                    #batch_labels.append(to_categorical(X_train_batch[i]['labels'], 28))
                    batch_labels[i][X_train_batch[i]['labels']] = 1

                yield np.array(batch_imgs, np.float32), batch_labels

    def label_one_hot(self, label):
        one_hot_label = np.zeros((28))
        #print(one_hot_label)
        for num in label:
            one_hot_label[num] = 1

        return one_hot_label