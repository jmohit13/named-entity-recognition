#!/bin/bash

mkdir glove_embedding && cd "$_" && \
wget http://nlp.stanford.edu/data/glove.6B.zip && \
unzip glove.6B.zip
rm glove.6B.zip