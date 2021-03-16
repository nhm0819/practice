# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 15:23:25 2021

@author: Hongmin
"""
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

# map
a = [1,2,3,4,5]
def double(in_val):
    return 2*in_val

doubled = list(map(double, a))
print(doubled)

# lambda lambda x : 2x
doubled2 = list(map(lambda x: 2*x, a))
print(doubled2)

# tf.cast
t1 = tf.constant([1,2,3,4,5])
t2 = tf.cast(t1, tf.float32)
print(t1.dtype)
print(t2.dtype)

#%%

train_ds = tfds.load(name='mnist',
                     shuffle_files = True,
                     as_supervised = True,
                     split = 'train',
                     batch_size = 4)

for images, labels in train_ds:
    print(images.shape, images.dtype)
    print(labels.shape, labels.dtype)
    print(tf.reduce_max(images))
    break

#%%

def standardization(images, labels):
    images = tf.cast(images, tf.float32) / 255.
    return [images, labels]

train_ds_iter = iter(train_ds)
images, labels = next(train_ds_iter)
print(images.dtype, tf.reduce_max(images))

train_ds = train_ds.map(standardization)
train_ds_iter = iter(train_ds)
images, labels = next(train_ds_iter)
print(images.dtype, tf.reduce_max(images))


#%%
def mnist_data_loader():
    
    def standardization(images, labels):
        images = tf.cast(images, tf.float32) / 255.
        return [images, labels]
    
    train_ds, test_ds = tfds.load(name='mnist',
                                  shuffle_files = True,
                                  as_supervised = True,
                                  split = ['train', 'test'],
                                  batch_size = 4)
    
    train_ds = train_ds.map(standardization)
    test_ds = test_ds.map(standardization)
    return train_ds, test_ds
