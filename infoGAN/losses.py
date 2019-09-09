from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.backend import dot
from tensorflow.linalg import tensor_diag_part
from math import log
import numpy as np

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        
def discriminatorLoss(real_output, fake_output, c_pred, batchSize):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    #real_loss = -tf.math.log(real_output)
    #real_loss = tf.math.reduce_mean(real_loss)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    #fake_loss = -tf.math.log(1-fake_output)
    #fake_loss = tf.math.reduce_mean(fake_loss)
    total_loss = real_loss + fake_loss
    return total_loss - Liqg(c_pred, batchSize)

def generatorLoss(fake_output, c_pred, batchSize):
    return -cross_entropy(tf.zeros_like(fake_output), fake_output) - Liqg(c_pred, batchSize)
    #fake_loss = tf.math.log(1-fake_output)
    #fake_loss = tf.math.reduce_mean(fake_loss)
    #return fake_loss - Liqg(c_pred)

def qLoss(c_pred, batchSize):
    return -Liqg(c_pred, batchSize)

def Liqg(c_pred, batchSize):
    #c_pred = tf.math.reduce_prod(c_pred, axis=1)
    #c = tf.linalg.matmul(c_pred, cForLoss)
    #c = tf.linalg.tensor_diag_part(c)
    #c = tf.reduce_sum(c)
    #c = tf.math.log(c)
    #l = 0.1*tf.math.reduce_mean(c) #- log(0.1) 
    #return l

    i = 0
    #c1 = tf.constant([])
    #c1 = []
    l = 0
    while i < batchSize:
        c = c_pred[10*i:10*(i+1), :]
        c = tf.linalg.tensor_diag_part(c)
        c = tf.math.log(c)
        c = 0.1*tf.math.reduce_sum(c)
        l = l + c
        #c1.append(tf.constant([c.eval()]))
        #if i == 0:
        #    c1 = c
        #else:
        #    c1 = tf.concat([c1, c], axis=0)
        i = i + 1
    #l = tf.math.reduce_mean(c1)
    l = l/float(batchSize)
    return l
