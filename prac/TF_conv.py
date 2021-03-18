# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 16:27:24 2021

@author: Hongmin
"""

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D

conv = Conv2D(filters=8, kernel_size=(3,3), padding='valid', activation='relu')
pool = MaxPooling2D(pool_size=(2,2), strides=2)

## p=(k-1)/2
## output_shape = (input + 2p - k)/s + 1

image = tf.random.normal(mean=0, stddev=1, shape=(1,28,28,1))
# print(image.shape)

conved = conv(image)
# print(conved.shape)

# print('w:', conv.get_weights()[0].shape)
# print('b:', conv.get_weights()[1].shape)

pooled = pool(conved)
print(pooled.shape)

#%%


conv1 = Conv2D(filters=8, kernel_size=(3,3), padding='valid', activation='relu')
pool1 = MaxPooling2D(pool_size=(2,2), strides=2)

conv2 = Conv2D(filters=16, kernel_size=(3,3), padding='valid', activation='relu')
pool2 = MaxPooling2D(pool_size=(2,2), strides=2)

conv3 = Conv2D(filters=32, kernel_size=(3,3), padding='valid', activation='relu')
pool3 = MaxPooling2D(pool_size=(2,2), strides=2)

print(image.shape)
conved = conv1(image)
print(conved.shape)
pooled = pool1(conved)
print(pooled.shape)
conved = conv2(pooled)
print(conved.shape)
pooled = pool2(conved)
print(pooled.shape)
conved = conv3(pooled)
print(conved.shape)
pooled = pool3(conved)
print(pooled.shape)

#%%

from tensorflow.keras.models import Sequential

model = Sequential()
model.add(Conv2D(filters=8, kernel_size=(3,3), padding='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2))

model.build(input_shape=(None,28,28,1))
model.summary()



