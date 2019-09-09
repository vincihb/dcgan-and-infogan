from __future__ import print_function, division

import tensorflow as tf

from tensorflow import GradientTape
import glob
import imageio
import os
import PIL
from tensorflow.keras import Sequential
from tensorflow.train import Checkpoint
from tensorflow.keras.layers import Dense, BatchNormalization, \
        LeakyReLU, Conv2DTranspose, Conv2D, Dropout, Flatten, Reshape
from tensorflow.keras.optimizers import Adam 
import time
import matplotlib.pyplot as plt
import numpy as np
from IPython import display
from losses import *

class DCGAN:

    def __init__(self):
        (self.trainIm, self.trainL), (_, _) = tf.keras.datasets.mnist.load_data()
        self.trainIm = self.trainIm.reshape(self.trainIm.shape[0], 28, 28, 1).astype('float32')
        self.trainIm = (self.trainIm - 127.5) / 127.5 # Normalize the images to [-1, 1]

        self.bufferSize = 60000
        self.batchSize = 256
        self.epochs = 50
        self.noiseDim = 100
        self.numOfGen = 16

        self.genOpt = Adam(1e-4)
        self.disOpt = Adam(1e-4)

        self.trainDataset = tf.data.Dataset.from_tensor_slices(self.trainIm).shuffle(self.bufferSize).batch(self.batchSize)
        
        self.generator = self.buildGenerator()
        self.discriminator = self.buildDiscriminator()

    def buildGenerator(self):
        model = Sequential()
        model.add(Dense(7*7*256, use_bias=False, input_shape=(100,)))
        model.add(BatchNormalization())
        model.add(LeakyReLU())

        model.add(Reshape((7, 7, 256)))
        assert model.output_shape == (None, 7, 7, 256) # Note: None is the batch size

        model.add(Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        assert model.output_shape == (None, 7, 7, 128)
        model.add(BatchNormalization())
        model.add(LeakyReLU())

        model.add(Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, 14, 14, 64)
        model.add(BatchNormalization())
        model.add(LeakyReLU())

        model.add(Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        assert model.output_shape == (None, 28, 28, 1)

        return model

    def buildDiscriminator(self):
        model = Sequential()
        model.add(Conv2D(64, (5, 5), strides=(2, 2), padding='same'))
        model.add(LeakyReLU())
        model.add(Dropout(0.3))

        model.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(LeakyReLU())
        model.add(Dropout(0.3))

        model.add(Flatten())
        model.add(Dense(1))

        return model

    @tf.function
    def trainStep(self, images):
        noise = tf.random.normal([self.batchSize, self.noiseDim])

        with GradientTape() as genTape, GradientTape() as disTape:
            generatedImages = self.generator(noise, training=True)

            realOutput = self.discriminator(images, training=True)
            fakeOutput = self.discriminator(generatedImages, training=True)

            genLoss = generatorLoss(fakeOutput)
            disLoss = discriminatorLoss(realOutput, fakeOutput)

        gradientsGenerator = genTape.gradient(genLoss, self.generator.trainable_variables)
        gradientsDiscriminator = disTape.gradient(disLoss, self.discriminator.trainable_variables)

        self.genOpt.apply_gradients(zip(gradientsGenerator, self.generator.trainable_variables))
        self.disOpt.apply_gradients(zip(gradientsDiscriminator, self.discriminator.trainable_variables))

    def train(self, epochs):
        dataset = self.trainDataset
        #self.saveCheck()
        for epoch in range(epochs):
            start = time.time()
            seed = tf.random.normal([self.numOfGen, self.noiseDim])

            for image_batch in dataset:
                self.trainStep(image_batch)

            # Produce images for the GIF as we go
            display.clear_output(wait=True)
            self.generateAndSaveImages(self.generator,
                             epoch + 1,
                             seed)

            print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
            #if (epoch+1)%5 == 0:
            #    self.checkpoint.save(file_prefix = self.checkpointPrefix)
        
        # Generate after the final epoch
        display.clear_output(wait=True)
        self.generateAndSaveImages(self.generator,
                           epochs,
                           seed)


    def generateAndSaveImages(self, model, epoch, test_input):
        # Notice `training` is set to False.
        # This is so all layers run in inference mode (batchnorm).
        predictions = model(test_input, training=False)

        fig = plt.figure(figsize=(4,4))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i+1)
            plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')

        plt.savefig('images/imageAtEpoch{:04d}.png'.format(epoch))

#    def saveCheck(self):
#        self.checkpointDir = './trainingCP'
#        self.checkpointPrefix = os.path.join(checkpointDir, "ckpt")
#        self.checkpoint = Checkpoint(generator_optimizer = self.genOpt
#                discriminator_optimizer = self.disOpt
#                generator = self.generator
#                discriminator = self.discriminator)

if __name__ == "__main__":
    dcgan = DCGAN()
    dcgan.train(500)
