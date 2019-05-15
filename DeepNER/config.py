#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 09 08:40:13 2019

@author: Jain, Mohit
"""


import os
import numpy as np

MODELS_PATH = './models/'
PLOT_IMG_DIR_ROOT = './reports/'
BEST_MODEL_FILE = 'best_model_path.npy'
ROOT_PATH = './data/'
# GLOVE_EMBEDDING_PATH = './glove_embedding/glove.6B.300d.txt'
GLOVE_EMBEDDING_PATH = './glove_embedding/glove.6B.50d.txt'
# GLOVE_EMBEDDING_PATH = './glove_embedding/glove.6B.200d.txt'

MAX_SEQUENCE_LENGTH = 150
MAX_FEATURES = 100000
EMBEDDING_DIM = 50
# EMBEDDING_DIM = 300
BATCH_SIZE = 50
NUM_EPOCHS = 10
ACTIVATION_FUNCTION = 'relu'
VALIDATION_SPLIT = 0.2
NUM_LSTM_UNITS = np.random.randint(150, 275)
NUM_DENSE_UNITS = np.random.randint(100, 150)
DROP_RATE_LSTM = 0.15 + np.random.rand() * 0.25
DROP_RATE_DENSE = 0.15 + np.random.rand() * 0.25

WORD_DICT_PATH = os.path.join(MODELS_PATH,'word_idx_lstm_%d_%d_%.2f_%.2f'%(NUM_LSTM_UNITS, NUM_DENSE_UNITS, DROP_RATE_LSTM, DROP_RATE_DENSE))
LABEL_DICT_PATH = os.path.join(MODELS_PATH,'label_idx_lstm_%d_%d_%.2f_%.2f'%(NUM_LSTM_UNITS, NUM_DENSE_UNITS, DROP_RATE_LSTM, DROP_RATE_DENSE))
PLOT_IMG_FILE = os.path.join(PLOT_IMG_DIR_ROOT,'lstm_%d_%d_%.2f_%.2f'%(NUM_LSTM_UNITS, NUM_DENSE_UNITS, DROP_RATE_LSTM, DROP_RATE_DENSE))
