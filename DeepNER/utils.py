#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 09 09:03:45 2019

@author: Jain, Mohit
"""

import numpy as np

from config import MAX_FEATURES,EMBEDDING_DIM,GLOVE_EMBEDDING_PATH

def get_embedding(embedding_type,
                  wrd2idx):
    """Generates embeddings and return them
    
    INPUT
    -----------
        embedding_type: type of embedding. word2vec, glove, fasttext
        wrd2idx: word index
        
    RETURN
    -----------
        embedding
    
    """
    if embedding_type == 'glove':
        vocab_size = MAX_FEATURES
        nb_words = min(vocab_size, len(wrd2idx)) + 1
        embedding_index = {}
        embedding_matrix = np.random.rand(nb_words,EMBEDDING_DIM)
        print("Embedding Shape {} Size {} ".format(embedding_matrix.shape, embedding_matrix.size))

        word_wth_embeddings = 0
        
        glove_vectors = open(GLOVE_EMBEDDING_PATH,'r')
        for line in glove_vectors:
            wrd = line.split()[0]
            coefs = np.asarray(line.split()[1:],dtype='float32')
            embedding_index[wrd] = coefs
        glove_vectors.close()
        print("Found {} glove vectors".format(len(embedding_index)))
        
        for word,idx in wrd2idx.items():
            if idx > vocab_size:
                continue
            embedding_vector = embedding_index.get(word)
            if embedding_vector is not None:
                word_wth_embeddings += 1
                embedding_matrix[idx] = embedding_vector
        
        return embedding_matrix, nb_words
    