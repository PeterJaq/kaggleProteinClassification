
import tensorflow as tf
from utils.DataLoad_K import DataSet
from model.InceptionV3 import Inception_V3
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import metrics
from tensorflow.keras import backend as K
import os
from PIL import Image


def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed

def focal_loss_new(gamma=2, alpha=0.25):

    def focal_loss_fixed(y_true, y_pred):
        eps = 1e-12
        y_pred=K.clip(y_pred,eps,1.-eps)
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
    
    return focal_loss_fixed


def f1_score(y_true, y_pred):
    def recall(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall
 
    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))



BATCH_SIZE = 16

Data_input = DataSet()
len_train = len(Data_input.train_data_index)
len_val = len(Data_input.val_data_index)
train_data = Data_input.train_generator()
val_data = Data_input.val_generator()

InceptionV3 = Inception_V3()
model = InceptionV3.create_model()

for layer in model.layers:
    layer.trainable = True

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=1e-4),
              metrics=['accuracy', metrics.categorical_crossentropy, focal_loss_new(gamma=2, alpha=0.25), f1_score])

model.summary()

model.fit_generator(train_data,
                    steps_per_epoch=int(float(len_train)/BATCH_SIZE),
                    validation_data=val_data,
                    validation_steps=int(float(len_val)/BATCH_SIZE),
                    epochs=100,
                    verbose=1,
                    class_weight='auto',
                    callbacks=InceptionV3.callbacks_list)


#model.fit_generator(Data_input.datagen.flow())
#model.fit(Data_input.make_one_shot_iterator(), epochs=10, steps_per_epoch=int(float(31072)/BATCH_SIZE))
model.save('./checkpoint/InceptionV3_B.h5')
