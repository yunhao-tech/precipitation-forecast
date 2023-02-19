import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential 
from keras.layers import ConvLSTM2D, BatchNormalization, LeakyReLU
from keras.layers.convolutional import Conv3D

import numpy as np
import matplotlib.pyplot as plt

class baseModel(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.conv1 = ConvLSTM2D(filters=64, kernel_size=(7, 7), 
                    input_shape=(18,344,315,1), 
                    padding='same',activation=LeakyReLU(alpha=0.01), return_sequences=True)
        self.norm = BatchNormalization(),
        self.conv2 = ConvLSTM2D(filters=64, kernel_size=(5, 5), 
                    padding='same',activation=LeakyReLU(alpha=0.01), return_sequences=True)
        self.conv3 = ConvLSTM2D(filters=64, kernel_size=(3, 3), 
                    padding='same',activation=LeakyReLU(alpha=0.01), return_sequences=True)
        self.conv4 = ConvLSTM2D(filters=64, kernel_size=(1, 1), 
                    padding='same',activation=LeakyReLU(alpha=0.01), return_sequences=True)
        self.conv5 = Conv3D(filters=1, kernel_size=(3, 3, 3), 
                activation='sigmoid', 
                padding='same', data_format='channels_last')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.norm(x)
        x = self.conv2(x)
        x = self.norm(x)
        x = self.conv3(x)
        x = self.norm(x)
        x = self.conv4(x)
        x = self.conv5(x) 


def get_estimator():
    model = baseModel()
    model.compile(loss='binary_crossentropy', optimizer='adadelta')
    return model