import tensorflow as tf


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


def create_network(X, Y, keep_prob, vocabulary, embedding_size, trained_embeddings=None):
    
    embedding = embedding_layer(X, len(vocabulary.index_to_word), embedding_size, trained_embeddings)    
    #conv_flat = tf.layers.flatten(embedding)
     
    conv1 = tf.layers.conv1d(inputs=embedding, filters=16, kernel_size=7, padding="same", activation=tf.nn.leaky_relu)
    pool1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=2, strides=2)

    #conv2 = tf.layers.conv1d(inputs=pool1, filters=32, kernel_size=7, padding="same", activation=tf.nn.leaky_relu)
    #pool2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=2, strides=2)
    
    conv_flat2 = tf.layers.flatten(pool1)

    fc1 = tf.layers.dense(inputs=conv_flat2, units=128, activation=tf.nn.leaky_relu)
    dropout1 = tf.nn.dropout(fc1, keep_prob)
    fc2 = tf.layers.dense(inputs=dropout1, units=64, activation=tf.nn.leaky_relu)
    dropout2 = tf.nn.dropout(fc2, keep_prob)
    #fc3 = tf.layers.dense(inputs=dropout2, units=32, activation=tf.nn.leaky_relu)
    #dropout3 = tf.nn.dropout(fc3, keep_prob)
    logits = tf.layers.dense(inputs=dropout2, units=2, activation=None)
        
    return logits