#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 09 09:12:06 2019

@author: Jain, Mohit
"""


import os
import numpy as np
from time import time
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Bidirectional
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras_contrib.layers import CRF

from utils import get_embedding


class DeepNER:
	""" NER with BiLSTM + CRF + Word Embeddings """
	def __init__(self,
		embedding_dim,
		activation_function,
		validation_split_ratio,
		number_epoch,
		batch_size,
		drop_rate_embedding,
		drop_rate_lstm,
		num_lstm_units):

		self.embedding_dim = embedding_dim
		self.activation_function = activation_function      
		self.validation_split_ratio = validation_split_ratio
		self.number_epoch = number_epoch
		self.batch_size = batch_size
		self.drop_rate_embedding = drop_rate_embedding        
		self.drop_rate_lstm = drop_rate_lstm
		self.num_lstm_units = num_lstm_units
		
	def train_model(self,
					X,
					y,
					labels,
					word_index,
					MAX_SEQUENCE_LENGTH,
					model_save_directory='./models/'):
		"""
			Train deep learning model
		"""
		
		embedding_matrix, nb_words = get_embedding('glove',word_index)
		
		input1 = Input(shape=(MAX_SEQUENCE_LENGTH,))
		embedding = Embedding(input_dim=len(embedding_matrix), 
								output_dim=self.embedding_dim, 
								weights=[embedding_matrix], 
								input_length=MAX_SEQUENCE_LENGTH, 
								trainable=False)(input1)
#         embedding = Dropout(self.drop_rate_embedding)(embedding)
		model = Bidirectional(LSTM(units=self.num_lstm_units,
								  return_sequences=True,
								  recurrent_dropout=self.drop_rate_lstm))(embedding)
		
		model = TimeDistributed(Dense(units=self.num_lstm_units,
									 activation=self.activation_function))(model)
		crf = CRF(units=len(labels))
		output1 = crf(model)
		
		model = Model(input1,output1)        
		model.compile(optimizer='rmsprop',\
					  loss=crf.loss_function,\
					 metrics=[crf.accuracy])
				
		print(model.summary())

		early_stopping = EarlyStopping(monitor='val_loss', patience=3)
		STAMP = 'lstm_%f_%.2f' % (self.num_lstm_units, self.drop_rate_lstm)
		checkpoint_dir = model_save_directory + 'checkpoints/' + str(int(time())) + '/'

		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)

		with open(bst_model_path+".json", "w") as json_file: json_file.write(model.to_json())
        model_checkpoint = ModelCheckpoint(bst_model_path+ '.h5', save_best_only=True, save_weights_only=False)
        print(bst_model_path+".json",bst_model_path+ '.h5')

		tensorboard = TensorBoard(log_dir=checkpoint_dir + "logs/{}".format(time()))

		history = model.fit(X,\
					np.array(y),\
					batch_size=self.batch_size,\
					epochs=self.number_epoch,\
					validation_split=self.validation_split_ratio,\
					callbacks=[early_stopping,model_checkpoint,tensorboard],\
					verbose=1)

		return history, model, bst_model_path
		