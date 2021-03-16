# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 14:12:51 2021

@author: Hongmin
"""
import os
import tensorflow as tf

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from utils.basic_utils import resetter, training_reporter
from utils.cp_utils import save_metrics_model, metric_visualizer
from utils.dataset_utils import load_processing_mnist, load_processing_cifar10
from utils.learning_env_setting import argparser, dir_setting, continue_setting, get_classification_metrics
from utils.train_validation_test import train, validation, test

from models import LeNet5

# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

''' =============== Learning Setting =============== '''
exp_name = 'learning_rate'
CONTINUE_LEARNING = False

train_ratio = 0.8
train_batch_size, test_batch_size = 32, 16

epochs = 30
save_period = 5
learning_rate = 0.01

exp_idx, epochs, learning_rate, train_batch_size, activation = argparser(epochs=epochs,
                                                                         learning_rate=learning_rate,
                                                                         train_batch_size=train_batch_size)

exp_name = 'exp' + str(exp_idx) + '_' + exp_name + '_LeNet5'
model = LeNet5()
optimizer = SGD(learning_rate = learning_rate)
''' =============== Learning Setting =============== '''

loss_object = SparseCategoricalCrossentropy()
path_dict = dir_setting(exp_name, CONTINUE_LEARNING)
model, losses_accs, start_epoch = continue_setting(CONTINUE_LEARNING, path_dict, model=model)
train_ds, validation_ds, test_ds = load_processing_mnist(train_ratio, train_batch_size, test_batch_size)
metric_objects = get_classification_metrics()


import time
start_time = time.time()

for epoch in range(start_epoch, epochs):
    train(train_ds, model, loss_object, optimizer, metric_objects)
    validation(validation_ds, model, loss_object, metric_objects)
    
    training_reporter(epoch, losses_accs, metric_objects, exp_name=exp_name)
    save_metrics_model(epoch, model, losses_accs, path_dict, save_period)
    
    metric_visualizer(losses_accs, path_dict['cp_path'])
    resetter(metric_objects)

end_time = time.time()
elapsed_time = end_time - start_time
with open(path_dict['cp_path'] + './elapsed_time.txt', 'w') as f:
    f.write(str(elapsed_time))

test(test_ds, model, loss_object, metric_objects, path_dict)