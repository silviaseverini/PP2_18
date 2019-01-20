import numpy as np

def convert_label(label_string):
	'''
	Labels:
	    "+" stands for "binding protein" => 1
	    "-" stands for "non-binding" => 0
	'''
	if label_string == "+":
		return 1
	elif label_string == "-":
		return 0
	else:
		print("Should not enter here")
		return None


def max_length_sentence(dataset):
    return max([len(line) for line in dataset])


def pad_sentence(tokenized_sentence, max_length_sentence, padding_value=0):
    
    pad_length = max_length_sentence - len(tokenized_sentence)
    sentence = list(tokenized_sentence)
    
    if pad_length > 0:
        return np.pad(tokenized_sentence, (0, pad_length), mode='constant', constant_values=int(padding_value))
    else: # Cut sequence if longer than max_length_sentence
        return sentence[:max_length_sentence]


def create_input(vocab, max_length_seq, sequences, labels):
    X, Y = [], []
    assert(len(sequences) == len(labels))
    
    for i in range(len(sequences)):
        X.append(pad_sentence(vocab.text_to_indices(sequences[i]), max_length_seq))
        Y.append(pad_sentence(labels[i], max_length_seq, padding_value=1))
        
    return np.array(X), np.array(Y)


def get_total_pos_neg(labels):
    total_pos = sum([sum(i) for i in labels])
    total_neg = sum([len(i) - sum(i) for i in labels])
    
    return total_pos, total_neg