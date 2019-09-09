from __future__ import print_function, division
  
import tensorflow as tf

from tensorflow import GradientTape, concat
from tensorflow.dtypes import cast
import glob
import imageio
import os
import PIL
from tensorflow.keras import Sequential, Model, Input
from tensorflow.train import Checkpoint
from tensorflow.keras.layers import Dense, BatchNormalization, \
        LeakyReLU, Conv2DTranspose, Conv2D, Dropout, Flatten, Reshape,\
        ReLU, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.random import normal
from tensorflow.keras.backend import random_uniform, concatenate, transpose, dot
from tensorflow.keras.utils import to_categorical
from tensorflow.linalg import tensor_diag_part
import time
import matplotlib.pyplot as plt
import numpy as np
from IPython import display

#generator_optimizer = tf.train.AdamOptimizer(1e-4)
#discriminator_optimizer = tf.train.AdamOptimizer(1e-4)

class TEST():
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.num_classes = 10
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.noiseDim = 100
        self.pltSize = 50
        (self.trainIm, self.trainL), (self.testIm, self.testL) = tf.keras.datasets.mnist.load_data()
        
        #self.genV = tf.random_normal([self.pltSize, self.noiseDim])


        self.trainIm = self.trainIm.reshape((60000, 28, 28, 1))
        self.testIm = self.testIm.reshape((10000, 28, 28, 1))
        
        self.trainIm = self.trainIm/255.0
        self.testIm = self.testIm/255.0

    def simpleClassification(self):
        model = Sequential()
        model.add(Flatten(input_shape = (28,28, 1)))
        #model.add(Dense(700, activation=LeakyReLU(alpha=0.3)))
        #model.add(Dense(400, activation=LeakyReLU(alpha=0.3)))
        model.add(Dense(350, activation=LeakyReLU(alpha=0.3)))
        #model.add(Dense(200, activation=LeakyReLU(alpha=0.3)))
        #model.add(Dense(100, activation=LeakyReLU(alpha=0.3)))
        #model.add(Dense(50, activation=LeakyReLU(alpha=0.3)))
        model.add(Dense(10, activation='softmax'))
        
        return model

    def simpleClass(self):
        model = Sequential()
        model.add(Flatten(input_shape= (28,28, 1)))
        model.add(Dense(10, activation='softmax'))
        return model
   
    def simpleTrain(self, epoch):
        self.simpleNN = self.simpleClassification()
        self.simpleNN.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
        self.simpleNN.fit(self.trainIm, self.trainL, epochs=epoch)

    def evaluateSimpleNN(self):
        test_loss, test_acc = self.CNN.evaluate(self.testIm, self.testL)
        print('Test accuracy:', test_acc)


    def convClass(self):
        model = Sequential()
        model.add(Conv2D(32, (3,3), padding='same', activation=LeakyReLU(alpha=0.3), input_shape=(28,28,1)))
        model.add(Conv2D(64, (3,3), activation=LeakyReLU(alpha=0.3)))
        #model.add(Conv2D(128, (3,3), activation=LeakyReLU(alpha=0.3)))
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
    infogan = TEST()
    #infogan.train(5)
    infogan.convTrain(5)
    #infogan.simpleTrain(10)
    infogan.evaluateSimpleNN()
    #infogan.train(epochs=5000, batch_size=128, sample_interval=50)
