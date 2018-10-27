import numpy as np
import tensorflow as tf
from tensorflow.data import Dataset
import pandas as pd
import os
from PIL import Image

class DataLoad:

    def __init__(self, batch_size=16):
        self.train_data_path = "../../data/preprocess/train"
        self.data = pd.read_csv("../data/train.csv")
        self.batch_size = batch_size

    def label_one_hot(self, label):
        one_hot_label = np.zeros((28))
        #print(one_hot_label)
        for num in label:
            one_hot_label[num] = 1

        return one_hot_label

    def load_csv_data(self):

        train_data_path = []
        train_data_label = []
        print(len(train_data_path))
        for name, labels in zip(self.data['Id'], self.data['Target'].str.split(' ')):
            train_data_path.append(os.path.join(self.train_data_path, name))
            train_data_label.append(self.label_one_hot(np.array([int(label) for label in labels])))

        train_data_path = np.array(train_data_path)
        train_data_label = np.array(train_data_label)
        dataset = Dataset.from_tensor_slices((train_data_path,
                                              train_data_label))
        dataset = dataset.map(self._parse_function)
        dataset = dataset.batch(self.batch_size)

        return dataset

    def _parse_function(self, picture_path, label):
        image_string = tf.read_file(picture_path+'.png')
        img_decode = tf.image.decode_png(image_string)
        img = tf.image.resize_images(img_decode, [299, 299])
        print(img, label)
        return img, label


