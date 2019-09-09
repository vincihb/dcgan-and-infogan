from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.backend import dot
from tensorflow.linalg import tensor_diag_part
from math import log

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        
def discriminatorLoss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generatorLoss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def qLoss(c_pred, c_true):
    #print(tf.shape(c_pred))
    #print(tf.shape(c_true))
    cTemp = dot(c_pred, c_true)
    c = tensor_diag_part(cTemp)
    return -0.1*cross_entropy(tf.ones_like(c), c) + log(0.1) 
