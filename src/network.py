import tensorflow as tf
import biovec
import numpy as np

def loadEmbeddings(vocab, filename, embedding_size):
    pv_model = biovec.models.load_protvec(filename)
    trained_embeddings = []
    
    for key in vocab.word_to_index.keys():
        if key != "<PAD>":
            trained_embeddings.append(pv_model[key])  
        else:
            trained_embeddings.append(np.random.uniform(-1.0, 1.0, embedding_size))      
        
    return np.array(trained_embeddings)


def new_weights(shape, name=None):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name=name)


def new_biases(length, name=None):
    return tf.Variable(tf.constant(0.1, shape=[length]), name=name)


def embedding_layer(input_x, vocabulary_size, embedding_size, trained_embeddings):

    # Train embeddings from scratch
    if trained_embeddings is None:
        init_embeds = tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0)
        embeddings = tf.Variable(init_embeds, name="embedding_weights")
        layer = tf.nn.embedding_lookup(embeddings, input_x)
    else:
        # Load pretrained embeddings
        embedding_weights = tf.get_variable(
            name='embedding_weights', 
            shape=(vocabulary_size, embedding_size), 
            initializer=tf.constant_initializer(trained_embeddings),
            trainable=True)

        layer = tf.nn.embedding_lookup(embedding_weights, input_x)
    
    return layer


def create_network(X, Y, keep_prob, vocabulary, embedding_size, hidden_cells, trained_embeddings=None, verbose=0):
    
    # Calculate length without padding
    mask_decoder_input = tf.cast(tf.sign(X), tf.float32)
    sequence_length = tf.cast(tf.reduce_sum(mask_decoder_input, 1), tf.int32)
    
    # Embedding layer
    embedding = embedding_layer(X, len(vocabulary.index_to_word), embedding_size, trained_embeddings)
    
    # Bidirectional LSTM cell
    lstm_fw_cell = tf.contrib.rnn.LSTMCell(hidden_cells)
    lstm_bw_cell = tf.contrib.rnn.LSTMCell(hidden_cells)

    # Dropout on LSTM cells
    dropout_fw = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell, input_keep_prob=keep_prob, output_keep_prob=keep_prob)
    dropout_bw = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell, input_keep_prob=keep_prob, output_keep_prob=keep_prob)
    
    (outputs_fw, outputs_bw), _ = tf.nn.bidirectional_dynamic_rnn(dropout_fw, 
                                                                  dropout_bw, 
                                                                  embedding, 
                                                                  dtype=tf.float32,
                                                                  sequence_length=sequence_length)
        
    # Concat outputs
    outputs_concat = tf.concat([outputs_fw, outputs_bw], 2)
    
    # FC layer
    fc1 = tf.layers.dense(inputs=outputs_concat, units=hidden_cells, activation=tf.nn.leaky_relu)
    logits = tf.layers.dense(inputs=fc1, units=2, activation=None)
    
    if verbose:
        print(embedding)
        print(outputs_concat)
        print(fc1)
        print(logits)
    
    return logits, mask_decoder_input, sequence_length
