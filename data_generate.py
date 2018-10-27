import cv2
import os
import pandas as pd
import numpy as np
from PIL import Image

def open_rgby(path, id): #a function that reads RGBY image
    colors = ['red', 'green', 'blue', 'yellow']
    flags = cv2.IMREAD_GRAYSCALE
    img = [cv2.imread(os.path.join(path, id) + '_'+color+'.png', flags) for color in colors]
    img[0] += (img[1] / 2).astype(np.uint8)
    img[2] += (img[3] / 2).astype(np.uint8)
    return np.stack((img[3], img[2], img[0]), axis=-1)

data = pd.read_csv("../data/sample_submission.csv")
train_data_path = "../data/test/"
train_data_path_list = []
save_path = "../data/perprocess/test/"
id_list = []
for name in data['Id']:
    train_data_path_list.append(name)
    id_list.append(name)
    """
        img = open_rgby(train_data_path,train_data_path_list[0])

    win = cv2.namedWindow('test win', flags=0)

    cv2.imshow('test win', img)

    cv2.waitKey(0)
    """
count = 0
for ids in train_data_path_list:
    count += 1
    img = open_rgby(train_data_path, ids)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    cv2.imwrite(os.path.join(save_path, ids)+".png", img)
    print("Count:%d ID:%s Saved in:%s" % (count, ids, os.path.join(save_path, ids)))