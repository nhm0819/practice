# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 19:58:21 2021

@author: Hongmin
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from termcolor import colored

import tensorflow as tf
import tensorflow_datasets as tfds
# tf.config.run_functions_eagerly(True)

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Flatten, Dense, Activation
# from tensorflow.keras.layers import InputLayer

os.environ['TF_FORCE_GUP_ALLOW_GROWTH'] = 'true'

#%% model build

model = Sequential()
# model.add(InputLayer=shape(28, 28, 1))
model.add(Flatten())
model.add(Dense(units=10, activation='relu'))
model.add(Dense(units=2))
model.add(Activation('softmax'))

model.build(input_shape=(None, 28, 28, 1))
# model.build()

model.summary()

# tf.keras.backend.clear_session()


#%%
class TestModel(Model):
    def __init__(self):
        super(TestModel, self).__init__()
        
        self.flatten = Flatten()
        self.d1 = Dense(units=10, activation='relu')
        self.d2 = Dense(units=2, activation='softmax')
        
    def call(self, x):
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        return x
    

model = TestModel()
model.build(input_shape=(None, 28, 28, 1))

model.summary()

     
#%%
model = Sequential()
model.add(Flatten())
model.add(Dense(units=10, activation='relu'))
model.add(Dense(units=2))
model.add(Activation('softmax'))

print(model.built)
test_img = tf.random.normal(shape=(1,28,28,1))
model(test_img)
print(model.built)


#%%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.layers import Flatten, Dense, Activation

model = Sequential()
model.add(Conv2D(filters=10, kernel_size=(3,3), padding='valid',
                 name='conv_1'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2,
                       name='conv_1_maxpool'))
model.add(Activation('relu', name='conv_1_act'))

model.add(Conv2D(filters=10, kernel_size=(3,3), padding='valid',
                 name='conv_2'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2,
                       name='conv_2_maxpool'))
model.add(Activation('relu', name='conv_2_act'))

model.add(Flatten(name='flatten_1'))
model.add(Dense(units=32, activation='relu',
          name='dense_1'))
model.add(Dense(units=10, activation='softmax',
          name='dense_2'))

model.build(input_shape=(None,28,28,1))

model.summary()

print(model.layers)
print(model.layers[0].get_weights())






