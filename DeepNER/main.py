#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 09 09:45:20 2019

@author: Jain, Mohit
"""

import os
import json

from config import MODELS_PATH,ROOT_PATH,WORD_DICT_PATH,LABEL_DICT_PATH,EMBEDDING_DIM,ACTIVATION_FUNCTION,VALIDATION_SPLIT,NUM_EPOCHS,BATCH_SIZE,DROP_RATE_DENSE,DROP_RATE_LSTM,NUM_LSTM_UNITS,MAX_SEQUENCE_LENGTH
from model import DeepNER
from data_prep import Dataset,get_train_test,get_sent_token_label


if __name__ == '__main__':

	if not os.path.exists(MODELS_PATH):
		os.makedirs(MODELS_PATH)

	TRAIN_PATH = os.path.join(ROOT_PATH,'train.txt')
	TEST_PATH = os.path.join(ROOT_PATH,'test.txt')

	labels_sent_list,tokens_sent_list,token_count,label_count,character_count = Dataset()._parse_dataset(TRAIN_PATH)
	sentences, tokens, labels = get_sent_token_label(tokens_sent_list,labels_sent_list,token_count,label_count)

	wrd2idx = {token: idx+1 for idx, token in enumerate(tokens)}
	
	with open(WORD_DICT_PATH+'.json', 'w') as fp:
		json.dump(wrd2idx, fp)
	print('Exported token dictionary')

	label2idx = {label:idx for idx, label in enumerate(labels)}
	with open(LABEL_DICT_PATH+'.json', 'w') as fp:
		json.dump(label2idx, fp)
	print('Exported label dictionary')

	X_train, X_test, y_train, y_test = get_train_test(wrd2idx,label2idx,sentences,labels)
		
	model = DeepNER(EMBEDDING_DIM,ACTIVATION_FUNCTION,VALIDATION_SPLIT,NUM_EPOCHS,BATCH_SIZE,DROP_RATE_DENSE,DROP_RATE_LSTM,NUM_LSTM_UNITS)
	
	model_fitted, bst_model_path = model.train_model(X_train,\
													X_test,\
													labels,\
													word_index=wrd2idx,\
													MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH)
