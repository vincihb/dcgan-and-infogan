from __future__ import print_function, division

import tensorflow as tf

from keras.datasets import mnist
from keras.layers import MaxPooling2D
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, concatenate
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils import to_categorical
import keras.backend as K

import matplotlib.pyplot as plt

import numpy as np

class INFOGAN():
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        (self.trainIm, self.trainL), (self.testIm, self.testL) = mnist.load_data()
        
        self.trainIm = self.trainIm.reshape((60000, 28, 28, 1))
        self.testIm = self.testIm.reshape((10000, 28, 28, 1))

        self.trainIm = self.trainIm/255.0
        self.testIm = self.testIm/255.0

    def buildGenerator(self):
        model = Sequential()

    def simpleClassification(self):
        model = Sequential()
        model.add(Flatten(input_shape = (28,28)))
        model.add(Dense(500, activation=LeakyReLU(alpha=0.3)))
        model.add(Dense(250, activation=LeakyReLU(alpha=0.3)))
        model.add(Dense(100, activation=LeakyReLU(alpha=0.3)))
        model.add(Dense(50, activation=LeakyReLU(alpha=0.3)))
        model.add(Dense(10, activation='softmax'))
        
        return model

    def simpleTrain(self, epoch):
        self.simpleNN = self.simpleClassification()
        self.simpleNN.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
        self.simpleNN.fit(self.trainIm, self.trainL, epochs=epoch)

    def evaluateSimpleNN(self):
        test_loss, test_acc = self.simpleNN.evaluate(self.testIm, self.testL)
        print('Test accuracy:', test_acc)


    def convClass(self):
        model = Sequential()
        model.add(Conv2D(32, (3,3), padding='same', activation=LeakyReLU(alpha=0.3), input_shape=(28,28,1)))
        model.add(Conv2D(64, (3,3), activation=LeakyReLU(alpha=0.3)))
        model.add(Conv2D(128, (3,3), activation=LeakyReLU(alpha=0.3)))
        model.add(Flatten())
        model.add(Dense(10, activation='softmax'))
        model.summary()
        
        return model

    def convClass1(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(10, activation='softmax'))

        return model

    def convTrain(self, epoch):
        self.CNN = self.convClass1()
        self.CNN.compile(optimizer = 'adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
        self.CNN.fit(self.trainIm, self.trainL, epochs=epoch)

if __name__ == '__main__':
    infogan = INFOGAN()
    infogan.convTrain(5)
    #infogan.simpleTrain(5)
    #infogan.evaluateSimpleNN()
    #infogan.train(epochs=5000, batch_size=128, sample_interval=50)
