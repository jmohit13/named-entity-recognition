#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 08 07:25:06 2019

@author: Jain, Mohit
"""



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




def prepare_data(tokens_sent_list,labels_sent_list,token_count,label_count):
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

