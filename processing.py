import numpy as np
import biovec


def create_input(sequences, labels, sliding_window_size, n_grams):
    X, Y = [], []

    # Iterates over all proteins in dataset
    for i in range(len(sequences)):

        # Loop over sequence
        for j in range(0, len(sequences[i]) - sliding_window_size + 1):
            sub_sequence = sequences[i][j:j+sliding_window_size]

            tmp = []
            for k in range(0, sliding_window_size - n_grams + 1):
                tmp.append(sub_sequence[k:k+n_grams])

            X.append(tmp)
            if not labels is None: # If I have labels, collect them as well
                Y.append(labels[i][j+(sliding_window_size//2)]) 

    return np.array(X), np.array(Y)


def loadEmbeddings(vocab, filename):
    pv_model = biovec.models.load_protvec(filename)
    trained_embeddings = []
    
    for key in vocab.word_to_index.keys():
        trained_embeddings.append(pv_model[key])        
        
    return np.array(trained_embeddings)
