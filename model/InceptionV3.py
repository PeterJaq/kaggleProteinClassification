import tensorflow
import numpy

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, GlobalMaxPooling2D, BatchNormalization, Input, Conv2D
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard


class Inception_V3:
    def __init__(self, batch_size=16):
        self.epochs = 100
        self.batch_size = batch_size
        self.checkpoint = ModelCheckpoint("./checkpoint/InceptionV3_B.h5", monitor='loss', verbose=1,
                                          save_best_only=True, mode='min', save_weights_only=False)
        self.reduceLROnPlat = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10,
                                           verbose=1, mode='auto', epsilon=0.0001)
        self.early = EarlyStopping(monitor="loss",
                              mode="auto",
                              patience=10)
        self.callbacks_list = [self.checkpoint, self.early, self.reduceLROnPlat, TensorBoard(log_dir='./tmp/log')]

    def create_model(self, input_shape=(299, 299, 3), n_out=28):

        input_tensor = Input(shape=input_shape)
        base_model = InceptionV3(include_top=False, weights="imagenet", input_shape=input_shape)
        #base_model = InceptionV3(, input_shape=input_shape)
        bn = BatchNormalization()(input_tensor)
        x = base_model(bn)
        x = Conv2D(16, kernel_size=(1, 1), activation='relu')(x)
        x = Flatten()(x)
        x = Dropout(0.5)(x)
        #x = BatchNormalization()(x)
        x = Dense(1024, activation='relu')(x)
        #x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        output = Dense(n_out, activation='sigmoid')(x)
        model = Model(input_tensor, output)

        return model