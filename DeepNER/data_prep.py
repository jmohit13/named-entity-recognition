#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 08 07:25:06 2019

@author: Jain, Mohit
"""

import collections
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from keras.utils import to_categorical

from config import MAX_SEQUENCE_LENGTH, VALIDATION_SPLIT

class Dataset(object):
    def __init__(self):
        pass
    
    def _parse_dataset(self, file_path):
        token_count = collections.defaultdict(lambda: 0)
        label_count = collections.defaultdict(lambda: 0)
        character_count = collections.defaultdict(lambda: 0)
        
        line_count = -1
        tokens_list = [];labels_list = []
        new_token_sequence = [];new_label_sequence = []
        
        if file_path:
            f = open(file_path, "r")
            for line in f:
                line_count += 1
                line = line.strip().split(' ')
                if len(line) == 0 or len(line[0]) == 0 or '-DOCSTART-' in line[0]:
                    if len(new_token_sequence) > 0:
                        labels_list.append(new_label_sequence)
                        tokens_list.append(new_token_sequence)
                        new_token_sequence = []; new_label_sequence = []
                    continue
                token = str(line[0]); label = str(line[-1])
                token_count[token] += 1; label_count[label] += 1
                
                new_token_sequence.append(token)
                new_label_sequence.append(label)
                for character in token: character_count[character] += 1
                
            if len(new_token_sequence) > 0:
                labels_list.append(new_label_sequence);tokens_list.append(new_token_sequence)
            f.close()
    
        return labels_list, tokens_list, token_count, label_count, character_count




def get_sent_token_label(tokens_sent_list,
    labels_sent_list,
    token_count,
    label_count):
    """
    Generates training data.
    
    INPUT
    ------------
    tokens_sent_list,labels_sent_list,token_count,label_count
    
    RETURN
    ------------
    SENTENCES : List od list of sentence words tuples
            [[('PRESS', 'O'),      ('DIGEST', 'O'),      ('-', 'O'),      ('Lebanon', 'B-LOC'),      ('-', 'O'),      ('Aug', 'O'),      ('22', 'O'),     ('.', 'O')],
             [('+1', 'O'),      ('Fred', 'B-PER'),      ('Couples', 'I-PER'),      ('through', 'O'),      ('15', 'O')]]

    tokens
            list of all words
    
    labels
            list of all labels
    """
    sentences=[]
    
    for list1,list2 in zip(tokens_sent_list,labels_sent_list):
        for idx,token_label in enumerate(zip(list1,list2)):
            if idx == 0:
                tmp = []
                tmp.append(token_label)
            else:
                tmp.append(token_label)
            sentences.append(tmp)

    sentences = [list(x) for x in set(tuple(x) for x in sentences)]  
    token_count_sorted = list(sorted(token_count.items(), key=lambda k_v: k_v[1],reverse=True))
    label_count_sorted = list(sorted(label_count.items(), key=lambda k_v: k_v[1],reverse=True))
    tokens = [i[0] for i in token_count_sorted]
    labels = [i[0] for i in label_count_sorted]
    
    return sentences, tokens, labels


def get_train_test(wrd2idx,
                   label2idx,
                   sentences,
                   labels):
    """
    Generates train and test data
    
    INPUT
    ----------
        wrd2idx
        label2idx
        sentences
        labels
    
    RETURN
    ----------
        train, test 
    """
    
    X = [[wrd2idx[tokn[0]] for tokn in sent] for sent in sentences]
    y = [[label2idx[label[1]] for label in sent] for sent in sentences]

    X_padded = sequence.pad_sequences(sequences=X,\
                                     maxlen=MAX_SEQUENCE_LENGTH,\
                                     padding='post')

    y_padded = sequence.pad_sequences(sequences=y,\
                                     maxlen=MAX_SEQUENCE_LENGTH,\
                                     padding='post')
    y_one_hot = [to_categorical(i, num_classes=len(labels)) for i in y_padded]
    
    X_tr, X_te, y_tr, y_te = train_test_split(X_padded, y_one_hot, test_size=VALIDATION_SPLIT)
    
    return X_tr, X_te, y_tr, y_te
