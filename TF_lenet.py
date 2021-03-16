# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 16:49:25 2021

@author: Hongmin
"""

import tensorflow as tf

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Conv2D, AveragePooling2D, ZeroPadding2D
from tensorflow.keras.layers import Flatten, Dense

# LeNet1 = Sequential()
# LeNet1.add(Conv2D(filters=4, kernel_size=(5,5), padding='valid', strides=1, activation='tanh'))
# LeNet1.add(AveragePooling2D(pool_size=(2,2), strides=2))
# LeNet1.add(Conv2D(filters=12, kernel_size=(5,5), padding='valid', strides=1, activation='tanh'))
# LeNet1.add(AveragePooling2D(pool_size=(2,2), strides=2))
# LeNet1.add(Flatten())
# LeNet1.add(Dense(units=10, activation='softmax'))

# LeNet1.build(input_shape=(None,28,28,1))
# LeNet1.summary()

#%%
# LeNet4 = Sequential()
# LeNet4.add(ZeroPadding2D(padding=(2,2)))
# LeNet4.add(Conv2D(filters=4, kernel_size=(5,5), padding='valid', strides=1, activation='tanh'))
# LeNet4.add(AveragePooling2D(pool_size=(2,2), strides=2))
# LeNet4.add(Conv2D(filters=16, kernel_size=(5,5), padding='valid', strides=1, activation='tanh'))
# LeNet4.add(AveragePooling2D(pool_size=(2,2), strides=2))
# LeNet4.add(Flatten())
# LeNet4.add(Dense(units=120, activation='tanh'))
# LeNet4.add(Dense(units=10, activation='softmax'))

# LeNet4.build(input_shape=(None,28,28,1))
# LeNet4.summary()


#%%
# LeNet5 = Sequential()
# LeNet5.add(ZeroPadding2D(padding=(2,2)))
# LeNet5.add(Conv2D(filters=6, kernel_size=(5,5), padding='valid', strides=1, activation='tanh'))
# LeNet5.add(AveragePooling2D(pool_size=(2,2), strides=2))
# LeNet5.add(Conv2D(filters=16, kernel_size=(5,5), padding='valid', strides=1, activation='tanh'))
# LeNet5.add(AveragePooling2D(pool_size=(2,2), strides=2))
# LeNet5.add(Flatten())
# LeNet5.add(Dense(units=140, activation='tanh'))
# LeNet5.add(Dense(units=84, activation='tanh'))
# LeNet5.add(Dense(units=10, activation='softmax'))

# LeNet5.build(input_shape=(None,28,28,1))
# LeNet5.summary()

#%% model subclassing

class FeatureExtractor(Layer):
    def __init__(self, filter1, filter2):
        super(FeatureExtractor, self).__init__()
        
        self.conv1 = Conv2D(filters=filter1, kernel_size=5, padding='valid', strides=1, activation='tanh')
        self.conv1_pool = AveragePooling2D(pool_size=2, strides=2)
        self.conv2 = Conv2D(filters=filter2, kernel_size=5, padding='valid', strides=1, activation='tanh')
        self.conv2_pool = AveragePooling2D(pool_size=2, strides=2)
        
    def call(self, x):
        x = self.conv1(x)
        x = self.conv1_pool(x)
        x = self.conv2(x)
        x = self.conv2_pool(x)
        return x


class LeNet1(Model):
    def __init__(self):
        super(LeNet1, self).__init__()
        
        # feature extractor
        self.conv1 = Conv2D(filters=4, kernel_size=(5,5), padding='valid', strides=1, activation='tanh')
        self.conv1_pool = AveragePooling2D(pool_size=(2,2), strides=2)
        self.conv2 = Conv2D(filters=12, kernel_size=(5,5), padding='valid', strides=1, activation='tanh')
        self.conv2_pool = AveragePooling2D(pool_size=(2,2), strides=2)
        
        # classifier
        self.flatten = Flatten()
        self.dense1 = Dense(units=10, activation='softmax')
    
    def call(self, x):
        x = self.conv1(x)
        x = self.conv1_pool(x)
        x = self.conv2(x)
        x = self.conv2_pool(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return x
    

# Subclassing + Sequential    
class LeNet4(Model):
    def __init__(self):
        super(LeNet4, self).__init__()
        
        ## feature extractor
        self.feature_extractor = Sequential()
        # self.zero_padding = ZeroPadding2D()
        # self.conv1 = Conv2D(filters=4, kernel_size=(5,5), padding='valid', strides=1, activation='tanh')
        # self.conv1_pool = AveragePooling2D(pool_size=(2,2), strides=2)
        # self.conv2 = Conv2D(filters=16, kernel_size=(5,5), padding='valid', strides=1, activation='tanh')
        # self.conv2_pool = AveragePooling2D(pool_size=(2,2), strides=2)
        self.feature_extractor.add(ZeroPadding2D(padding=2))
        self.feature_extractor.add(Conv2D(filters=4, kernel_size=(5,5), padding='valid', strides=1, activation='tanh'))
        self.feature_extractor.add(AveragePooling2D(pool_size=(2,2), strides=2))
        self.feature_extractor.add(Conv2D(filters=16, kernel_size=(5,5), padding='valid', strides=1, activation='tanh'))
        self.feature_extractor.add(AveragePooling2D(pool_size=(2,2), strides=2))
        
        ## classifier
        self.classifier = Sequential()
        # self.flatten = Flatten()
        # self.dense1 = Dense(units=120, activation='tanh')
        # self.dense2 = Dense(units=10, activation='softmax')
        self.classifier.add(Flatten())
        self.classifier.add(Dense(units=120, activation='tanh'))
        self.classifier.add(Dense(units=10, activation='softmax'))
    
    def call(self, x):
        # x = self.zero_padding(x)
        # x = self.conv1(x)
        # x = self.conv1_pool(x)
        # x = self.conv2(x)
        # x = self.conv2_pool(x)
        # x = self.flatten(x)
        # x = self.dense1(x)
        # x = self.dense2(x)
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x

# Featuer Extractor classing
class LeNet5(Model):
    def __init__(self):
        super(LeNet5, self).__init__()
        
        # feature extractor
        self.zero_padding = ZeroPadding2D(padding=2)
        self.feature_extractor = FeatureExtractor(6, 16)
        
        # classifier
        self.classifier = Sequential()
        self.classifier.add(Flatten())
        self.classifier.add(Dense(units=140, activation='tanh'))
        self.classifier.add(Dense(units=84, activation='softmax'))
        self.classifier.add(Dense(units=10, activation='tanh'))
    
    def call(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x

