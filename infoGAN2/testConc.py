from __future__ import print_function, division
import tensorflow as tf
from tensorflow.random import normal
from tensorflow.keras.backend import random_uniform, concatenate, transpose, dot
from tensorflow.keras.utils import to_categorical

a = random_uniform(shape=(15,1), minval=0, maxval=9, dtype='int32')
d = to_categorical(a, num_classes=10)
d = transpose(d)
b = normal([15, 4])
a = tf.dtypes.cast(a, dtype='float')
c = tf.concat([a, b], axis = 1)
print(a-1)
print(b.numpy())
print(c.numpy())
print(d)
eM = normal([15, 10])
print(eM)
print(tf.linalg.tensor_diag_part(dot(eM, d)))
