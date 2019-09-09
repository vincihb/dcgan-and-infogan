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
        ReLU
from tensorflow.keras.optimizers import Adam 
from tensorflow.random import normal
from tensorflow.keras.backend import random_uniform, concatenate, transpose, dot
from tensorflow.keras.utils import to_categorical
from tensorflow.linalg import tensor_diag_part
import time
import matplotlib.pyplot as plt
import numpy as np
from IPython import display
from losses import *
import math

class DCGAN:

    def __init__(self):
        (self.trainIm, self.trainL), (_, _) = tf.keras.datasets.mnist.load_data()
        self.trainIm = self.trainIm.reshape(self.trainIm.shape[0], 28, 28, 1).astype('float32')
        self.trainIm = (self.trainIm - 127.5) / 127.5 # Normalize the images to [-1, 1]

        self.bufferSize = 60000
        self.batchSize = 128
        self.epochs = 50
        self.noiseDim = 100
        self.numOfGen = 100
        self.numClasses = 10

        self.genOpt = Adam(1e-3, 0.5)
        self.disOpt = Adam(2e-4, 0.5)
        self.qNetOpt = Adam(2e-4,0.5)

        
        self.generator = self.buildGenerator()
        (self.discriminator, self.qNet) = self.buildDiscriminator()

        self.latentC = []
        i = 0
        while i < self.numOfGen:
            self.latentC.append([i%10])
            i = i+1 
        self.latentC = tf.constant(self.latentC, dtype='float')
        self.latentC = to_categorical(self.latentC, num_classes=10)
        #print(self.latentC)

        i = 0 
        self.latentCodes = []
        while i < 10:
            self.latentCodes.append([i%10])
            i = i + 1
        self.latentCodes = tf.constant(self.latentCodes, dtype = 'float')
        self.latentCodes = to_categorical(self.latentCodes, num_classes=10)

    def buildGenerator(self):
        model = Sequential()
        model.add(Dense(7*7*256, use_bias=False, input_shape=(100,)))
        model.add(BatchNormalization())
        model.add(ReLU())

        model.add(Reshape((7, 7, 256)))
        assert model.output_shape == (None, 7, 7, 256) # Note: None is the batch size

        model.add(Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        assert model.output_shape == (None, 7, 7, 128)
        model.add(BatchNormalization())
        model.add(ReLU())

        model.add(Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, 14, 14, 64)
        model.add(BatchNormalization())
        model.add(ReLU())

        model.add(Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        assert model.output_shape == (None, 28, 28, 1)

        return model

    def buildDiscriminator(self):
        img1 = Input(shape=(28, 28, 1))

        model = Sequential()
        model.add(Conv2D(64, (4, 4), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(0.1))
        model.add(Dropout(0.3))

        model.add(Conv2D(128, (4, 4), strides=(2, 2), padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(0.1))
        model.add(Dropout(0.3))

        img2 = model(img1)
        
        img3 = Flatten()(img2)
        #validity = Dense(1, activation='sigmoid')(img3)
        validity = Dense(1)(img3)

        qNet1 = Flatten()(img2)
        qNet2 = Dense(128, activation=LeakyReLU(0.1))(qNet1)
        qNet3 = BatchNormalization()(qNet2)
        label = Dense(self.numClasses, activation='softmax')(qNet3)

        return (Model(img1, validity), Model(img1, label))

    def noiseAndLatentC(self):
        noise = normal([self.batchSize, self.noiseDim - 10])
        rC = random_uniform(shape=(1, 1), minval=0, maxval=9, dtype='int32')
        rC = rC*np.ones(shape=(self.batchSize, 1))
        cForLoss = to_categorical(rC, num_classes=10)
        cForLoss = cast(cForLoss, dtype='float')
        noiseAndC = concat([cForLoss, noise], axis =1)
        cForLoss = transpose(cForLoss)
        #print(noiseAndC)
        #rC = cast(rC, dtype='float')
        #noiseAndC = concat([rC, noise], axis=1)
        return (noiseAndC, cForLoss)

    def noiseAndC(self):
        noise = normal([self.batchSize, self.noiseDim - 10])
        #print(noise[3])
        nC = []
        i = 0
        while i < self.batchSize:
            j = 0
            while j < 10:
                nC.append(concat([self.latentCodes[j], noise[i]], axis=-1).numpy())
                j = j + 1
            i = i + 1
        nC = tf.constant(nC)
        return nC
        #print(nC[3])
        #wait()

    @tf.function
    def trainStep(self, images, noiseAndC):
        with GradientTape() as genTape, GradientTape() as disTape, GradientTape() as cTape:
            generatedImages = self.generator(noiseAndC, training=True)

            realOutput = self.discriminator(images, training=True)
            fakeOutput = self.discriminator(generatedImages, training=True)
            cApprox = self.qNet(generatedImages, training=True)

            genLoss = generatorLoss(fakeOutput, cApprox, self.batchSize)
            disLoss = discriminatorLoss(realOutput, fakeOutput, cApprox, self.batchSize)
            qNetLoss = qLoss(cApprox, self.batchSize)
        
        gradientsGenerator = genTape.gradient(genLoss, self.generator.trainable_variables)
        gradientsDiscriminator = disTape.gradient(disLoss, self.discriminator.trainable_variables)
        gradientsQnet = cTape.gradient(qNetLoss, self.qNet.trainable_variables)

        self.genOpt.apply_gradients(zip(gradientsGenerator, self.generator.trainable_variables))
        self.disOpt.apply_gradients(zip(gradientsDiscriminator, self.discriminator.trainable_variables))
        self.qNetOpt.apply_gradients(zip(gradientsQnet, self.qNet.trainable_variables))
        #return realOutput, fakeOutput, cApprox, qNetLoss, genLoss, disLoss
        return qNetLoss

    def train(self, epochs):
        #self.saveCheck()
        Li = []
        L2 = []
        for epoch in range(epochs):
            start = time.time()
            dataset = tf.data.Dataset.from_tensor_slices(self.trainIm).shuffle(self.bufferSize).batch(self.batchSize)
            for image_batch in dataset:
                #(noiseAndC, cForLoss) = self.noiseAndLatentC()
                noiseAndC = self.noiseAndC()
                #(realOutput, fakeOutput, cApprox , qNetLoss, genLoss, disLoss) = self.trainStep(image_batch, noiseAndC)
                qNetLoss = self.trainStep(image_batch, noiseAndC)
                #print(qNetLoss)
                #print(cForLoss)
                #c = tf.linalg.matmul(cApprox, cForLoss)
                #c = tf.linalg.tensor_diag_part(c)
                #print(c)
                L = -qNetLoss.numpy()-math.log(0.1)
                #print("===============")
                #print(qNetLoss + log(0.1))
                #print(genLoss)
                #print(disLoss)
                #print(fakeOutput[0])
                #fake_loss = tf.math.log(1-fakeOutput)
                #print(fake_loss[0])
                #fake_loss = tf.math.reduce_mean(fake_loss)
                #print(fake_loss + qNetLoss)
                break
            print("=============")
            print(L)
            Li.append(L)
            L2.append(-math.log(0.1))

            #display.clear_output(wait=True)
            #self.generateAndSaveImages(self.generator,
            #                 epoch + 1)

            print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
            if ((epoch)%10 == 0) and (epoch < 1001):
                plt.plot(Li)
                plt.plot(L2)
                plt.xlabel("Epochs")
                plt.ylabel("Variational lower bound, L(g, q)")
                plt.grid(True)
                plt.savefig('plots/plot{:04d}.png'.format(epoch))
                plt.close()

            if ((epoch)%50 == 0):
                display.clear_output(wait=True)
                self.generateAndSaveImages(self.generator,
                        epoch + 1)
            #    self.checkpoint.save(file_prefix = self.checkpointPrefix)
        
        # Generate after the final epoch
        display.clear_output(wait=True)
        self.generateAndSaveImages(self.generator,
                           epochs)
        plt.plot(Li)
        plt.grid(True)
        plt.savefig('plots/plot.png')
        plt.close()

    def generateAndSaveImages(self, model, epoch):
        # Notice `training` is set to False.
        # This is so all layers run in inference mode (batchnorm).
        noise = tf.random.normal([self.numOfGen, self.noiseDim-10])
        noiseAndC = concat([self.latentC, noise], axis=1)

        #print(noiseAndC)

        predictions = model(noiseAndC, training=False)

        fig = plt.figure(figsize=(10,10))

        for i in range(predictions.shape[0]):
            plt.subplot(10, 10, i+1)
            plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')

        plt.savefig('images/imageAtEpoch{:04d}.png'.format(epoch))
        plt.close()

#    def saveCheck(self):
#        self.checkpointDir = './trainingCP'
#        self.checkpointPrefix = os.path.join(checkpointDir, "ckpt")
#        self.checkpoint = Checkpoint(generator_optimizer = self.genOpt
#                discriminator_optimizer = self.disOpt
#                generator = self.generator
#                discriminator = self.discriminator)

if __name__ == "__main__":
    dcgan = DCGAN()
    dcgan.train(150000)
