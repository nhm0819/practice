# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 17:33:04 2021

@author: Hongmin
"""

import tensorflow_datasets as tfds
dataset, ds_info = tfds.load(name='mnist',
                             shuffle_files = True,
                             with_info = True)

print(ds_info)
print(ds_info.splits)
print(dir(ds_info))
print(dataset.keys(), '\n')
print(dataset.values())

train_ds = dataset['train'].batch(32)
test_ds = dataset['test']
print(type(train_ds))

EPOCHS = 10
for epoch in EPOCHS:
    for data in train_ds:
        images = data['image']
        labels = data['label']

for tmp in train_ds:
    print(type(tmp))
    print(tmp.keys())
    images = tmp['image']
    labels = tmp['label']
    print(images.shape)
    print(labels.shape)
    break

#%%
dataset, ds_info = tfds.load(name='mnist',
                             shuffle_files = True,
                             with_info = True,
                             as_supervised = True)

train_ds = dataset['train'].batch(32)
test_ds = dataset['test']
print(type(train_ds))

for images, labels in train_ds:
    print(images.shape)
    print(labels.shape)
    break


#%% 
(train_ds, test_ds), ds_info = tfds.load(name='mnist',
                                         shuffle_files = True,
                                         as_supervised = True,
                                         split = ['train','test'],
                                         with_info = True)

#%%
# dataset, ds_info = tfds.load(name='patch_camelyon',
#                              shuffle_files = True,
#                              as_supervised = True,
#                              with_info = True)

print(ds_info.features)
print(ds_info.splits)

# (train_ds, validation_ds, test_ds), ds_info = tfds.load(name='patch_camelyon',
#                                                         shuffle_files = True,
#                                                         as_supervised = True,
#                                                         split = ['train', 'validation', 'test']
#                                                         with_info = True,
#                                                         batch_size = 9)
train_ds = train_ds.shuffle(10000)
train_ds_iter = iter(train_ds)
images, labels = next(train_ds_iter)
images = images.numpy()
labels = labels.numpy()
print(images.shape)
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3,3,figsize=(15,15))

for ax_idx, ax in enumerate(axes.flat):
    ax.imshow(images[ax_idx, ...])
    ax.set_title(labels[ax_idx], fontsize=15)
    
    ax.get_xaxis().set.visible(False)
    ax.get_yaxis().set.visible(False)

