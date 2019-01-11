#!/bin/bash

MODELS_FOLDER="./models/"

for ngram_size in 1 2 3 4 5
do
	model_path=$MODELS_FOLDER"ngram-"$ngram_size".model"

	# Generate corpus first
	python train_word2vec.py -i ./uniprot_sprot.fasta -n $ngram_size
	
	# Generate model
	python train_word2vec.py -c ./corpus.txt -o $model_path

	# Delete current corpus
	rm ./corpus.txt
done 
