#!/bin/bash

MODELS_FOLDER="./models/"

# Generate corpus first
python train_word2vec.py -i ./uniprot_sprot.fasta

for ngram_size in 1 2 3 4 5
do
	model_path=$MODELS_FOLDER"ngram-"$ngram_size".model"
	
	python train_word2vec.py -c ./corpus.txt -o $model_path -n $ngram_size
done 
