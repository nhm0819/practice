# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 11:53:53 2021

@author: Hongmin
"""

# import numpy as np
# a = np.load('./LeNet5_train1/losses_accs.npz')
# dir(a)
# a['validation_losses']
# a.close()
import argparse
import os
def argparser(epochs, learning_rate, train_batch_size):
    parser = argparse.ArgumentParser(description='hyperparameters for training')
    
    parser.add_argument('-e', type=int, default=None,
                        help='an integer for epochs')
    parser.add_argument('-l', type=float, default=None,
                        help='an floating point for learning rate')
    parser.add_argument('-b', type=int, default=None,
                        help='an integer for batch size')
    parser.add_argument('-a', type=str, default=None,
                        help='an string for activation function')
    parser.add_argument('-c', type=int, default=None,
                        help='an integer for experiment count')
    
    args = parser.parse_args()

    if args.e == None:
        epochs = epochs
    else:
        epochs = args.e
    if args.l == None:
        learning_rate = learning_rate
    else:
        learning_rate = args.l
    if args.b == None:
        train_batch_size = train_batch_size
    else:
        train_batch_size = args.b
    if args.a == None:
        activation = 'relu'
    else:
        activation = args.a
    if args.c == None:
        exp_idx = 0
    else:
        exp_idx = args.c
                
    return exp_idx, epochs, learning_rate, train_batch_size, activation

print(argparser(10, 0.01, 32))