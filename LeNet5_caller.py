# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 14:35:59 2021

@author: Hongmin
"""

import os

learning_rates = [0.01, 0.03, 0.1, 0.3]
n_exp = len(learning_rates)

for exp_cnt, exp_idx in enumerate(range(10, 10+n_exp)):
    learning_rate = learning_rates[exp_cnt]
    
    command_template = 'python ./LeNet5.py' + ' -l ' + \
        str(learning_rate) + ' -c ' + str(exp_idx)
    os.system(command_template)