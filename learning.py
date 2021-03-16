# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 13:34:32 2021

@author: Hongmin
"""

'''
-- project directory
    -- main.py
    -- models.py
    -- utils.py
        -- basic_utils.py
        -- cp_utils.py
        -- dataset_utils.py
        -- learning_env_setting.py
        -- train_validation_test.py
        
    -- train1
        -- confusion_matrix
        -- model
        -- losses_accs.npz
        -- losses_accs_visualization.png
        -- test_result.txt
'''

import os
import tensorflow as tf
import numpy as np
from termcolor import colored

from utils.learning_env_setting import dir_setting, continue_setting, get_classification_metrics
from utils.dataset_utils import load_processing_mnist

dir_name = 'train1'
CONTINUE_LEARNING = True

path_dict = dir_setting(dir_name, CONTINUE_LEARNING)
model = 'test'
model, losses_accs, start_epoch = continue_setting(CONTINUE_LEARNING, path_dict, model)









