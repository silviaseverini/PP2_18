import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import src.dictionary as dictionary
import src.processing as processing
import src.network as network
import pickle
import argparse

parser = argparse.ArgumentParser(description='Generate your predictions')
parser.add_argument('-ng', help='N grams', required=True)
parser.add_argument('-sws', help='Sliding window size', required=True)
parser.add_argument('-es', help='Embedding size', required=True)
parser.add_argument('-ra', help='Regularisation alpha', required=True)
parser.add_argument('-lr', help='Learning rate', required=True)
parser.add_argument('-dp', help='Dropout probability', required=True)
parser.add_argument('-bs', help='Batch size', required=True)

args = vars(parser.parse_args())

# Parameters
n_grams = int(args['ng'])
sliding_window_size = int(args['sws'])
embedding_size = int(args['es'])
reg_alpha = float(args['ra'])
lr = float(args['lr'])
dropout_prob = float(args['dp'])
batch_size = int(args['bs'])

use_pretrained_embeddings = embedding_size == 100
epochs = 1000

current_model_name = "model-{}-{}-{}-{}-{}-{}-{}".format(
  n_grams,
  sliding_window_size,
  embedding_size,
  reg_alpha,
  lr,
  dropout_prob,
  batch_size
)

print("Model {}".format(current_model_name))

assert(n_grams % 2 == 1 and sliding_window_size % 2 == 1)

# Save parameters, so that when restore model for testing, I have them
parameters = {"n_grams" : n_grams, 
              "sliding_window_size" : sliding_window_size, 
              "embedding_size" : embedding_size
             }

# Prepare dataset for prediction
protein_names, sequences, labels = [], [], []

'''
    Labels:
        "+" stands for "binding protein" => 1
        "-" stands for "non-binding" => 0
'''
def convert_label(label_string):
 
    if label_string == "+":
        return 1
    elif label_string == "-":
        return 0
    else:
        print("Should not enter here")
        return None
  
# Open file containing dataset    
with open('./dataset/ppi_data.fasta') as f:
    lines = f.read().splitlines()
    
    for i in range(len(lines)):
        
        if i % 3 == 0:
            protein_names.append(lines[i])
        elif i % 3 == 1:
            sequences.append(lines[i])
        elif i % 3 == 2:
            labels.append([convert_label(letter) for letter in lines[i]])
            
protein_names = np.array(protein_names)
sequences = np.array(sequences)
labels = np.array(labels)

assert(protein_names.shape[0] == sequences.shape[0] == labels.shape[0])

# print(protein_names[0])
# print(sequences[0])
# print(labels[0])

#sequences = sequences[:24]
#labels = labels[:24]

# Split percentage of training and validation
split_percentage = 0.8

# Count how many samples into training dataset
total_dataset = len(sequences)
train_dataset = int(total_dataset * split_percentage)

# Shuffle
np.random.seed(97)
indices = list(range(total_dataset))
np.random.shuffle(indices)

# Train dataset
sequences_train = sequences[indices[:train_dataset]]
labels_train = labels[indices[:train_dataset]]

# Validation dataset
sequences_val = sequences[indices[train_dataset:]]
labels_val = labels[indices[train_dataset:]]

# Shapes
# print("Training samples: " + str(sequences_train.shape[0]))
# print("Validation samples: " + str(sequences_val.shape[0]) + "\n")

# Reset seed for randomness
np.random.seed()


X_train, Y_train = processing.create_input(sequences_train, labels_train, sliding_window_size, n_grams)
X_val_proc, Y_val_proc = processing.create_input(sequences_val, labels_val, sliding_window_size, n_grams)

# print(X_train.shape)
# print(X_val_proc.shape)

# print(X_train[0])
# print(Y_train[0])

# Save dump for parameters
parameters["timestamps"] = X_train.shape[1]
with open('./dumps/parameters.pickle', 'wb') as handle:
    pickle.dump(parameters, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Find indices with positive and negative labels
X_train_pos = np.array([i for i in range(len(X_train)) if Y_train[i] == 1])
X_train_neg = np.array([i for i in range(len(X_train)) if Y_train[i] == 0])

# Print how many positive and negative labels => I want same number of labels for each class during training
# print("Positive: " + str(X_train_pos.shape[0]) + " | Negative: " + str(X_train_neg.shape[0]))

# Get indices from X_train_pos
np.random.seed(97)
X_train_pos_indices = np.random.choice(len(X_train_pos), len(X_train_neg), replace=True)
X_train_selected = X_train_pos[X_train_pos_indices]

# Final X_train data
X_train = np.concatenate((X_train[X_train_selected], X_train[X_train_neg]), axis=0)
Y_train = np.concatenate((Y_train[X_train_selected], Y_train[X_train_neg]), axis=0)

# Check that labels 1 and 0 are equal
assert(len(np.array([i for i in range(len(X_train)) if Y_train[i] == 1])) == 
    len(np.array([i for i in range(len(X_train)) if Y_train[i] == 0])))

# print("Final training data shape: " + str(X_train.shape))

# Create vocabulary of n-grams
vocab = dictionary.LanguageDictionary(X_train)

# Load embeddings trained with Protovec
trained_embeddings = processing.loadEmbeddings(vocab, "./train_embeddings/models/ngram-" + str(n_grams) + ".model")

assert(len(vocab.index_to_word) == len(vocab.word_to_index) == len(trained_embeddings))
# print("Embedding layer words: " + str(len(trained_embeddings)))

# Save vocabulary locally with pickle dump
with open('./dumps/vocab.pickle', 'wb') as handle:
    pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Map grams to indices for the embedding matrix and remove samples where unknown words
X_train = np.array([vocab.text_to_indices(tmp) for tmp in X_train])
# print(X_train.shape)

# Prepare validation data
X_val, Y_val = [], []
for i in range(len(X_val_proc)):
    
    tmp = vocab.text_to_indices(X_val_proc[i])
    if not None in tmp:
        X_val.append(tmp)
        Y_val.append(Y_val_proc[i])
        
X_val = np.array(X_val)
Y_val = np.array(Y_val)

assert(len(X_val) == len(Y_val))
# print(X_val.shape)


tf.reset_default_graph()

# Placeholders
tensor_X = tf.placeholder(tf.int32, (None, X_train.shape[1]), 'inputs')
tensor_Y = tf.placeholder(tf.int32, (None), 'output')

keep_prob = tf.placeholder(tf.float32, (None), 'dropout_input')

# Create graph for the network
if use_pretrained_embeddings:
    assert(len(trained_embeddings[0]) == embedding_size)
else:
    trained_embeddings = None

logits = network.create_network(tensor_X, tensor_Y, keep_prob, vocab, embedding_size, trained_embeddings)

# Cross entropy loss after softmax of logits
ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tensor_Y)
meaned = tf.reduce_mean(ce)

trainable_vars = tf.trainable_variables()
l2_reg = tf.reduce_sum([tf.nn.l2_loss(var) for var in trainable_vars])
loss = meaned + reg_alpha * l2_reg

# Using Adam (Adaptive learning rate + momentum) for the update of the weights of the network
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

# Useful tensors
scores = tf.nn.softmax(logits)
predictions = tf.to_int32(tf.argmax(scores, axis=1))
correct_mask = tf.to_float(tf.equal(predictions, tensor_Y))
accuracy = tf.reduce_mean(tf.cast(correct_mask, tf.float32), axis=0)

# Training data variables
iterations_training = max((len(X_train) // batch_size), 1)
# print("Training iterations per epoch: " + str(iterations_training))

# Validation data variables
max_val_acc = 0
iterations_validation = max((len(X_val) // batch_size), 1)

# Perform each epoch, shuffle training dataset
indices = list(range(len(X_train)))
consecutive_validation = 0

saver = tf.train.Saver()
with tf.Session() as sess:
    
    # Initialize variables in the graph
    sess.run(tf.global_variables_initializer())
    
    # Iterate over epochs
    for i in range(epochs):
        
        # Shuffle data (with random seed for debug) to not train the network always with the same order
        np.random.seed(97)
        np.random.shuffle(indices)
        X_train = X_train[indices]
        Y_train = Y_train[indices]
        
        # Vector accumulating accuracy and loss during one epoch
        total_accuracies, total_losses = [], []

        # Iterate over mini-batches
        for j in range(iterations_training):
            start_index = j * batch_size
            end_index = (j + 1) * batch_size 
            
            # If last batch, take also elements that are less than batch_size
            if j == (iterations_training - 1):
                end_index += (batch_size - 1)

            _, avg_accuracy, avg_loss = sess.run([optimizer, accuracy, loss], feed_dict={
                                                        tensor_X: X_train[start_index:end_index],
                                                        tensor_Y: Y_train[start_index:end_index],
                                                        keep_prob: dropout_prob})
            # Add values for this mini-batch iterations
            total_losses.append(avg_loss) 
            total_accuracies.append(avg_accuracy)

            # Statistics on validation set
            if (j+1) % 30 == 0:  
                avg_accuracy, avg_loss, pred = sess.run([accuracy, loss, predictions], feed_dict={ 
                                                                                    tensor_X: X_val,
                                                                                    tensor_Y: Y_val,
                                                                                    keep_prob: 1.0 })
                #avg_accuracy = precision_score(Y_val, pred)
                
                # Save model if validation accuracy better
                if avg_accuracy > max_val_acc:
                    consecutive_validation_without_saving = 0
                    max_val_acc = avg_accuracy
                    #print("SAVE | Val loss: " + str(avg_loss) + ", accuracy: " + str(avg_accuracy))
                    save_path = saver.save(sess, "./checkpoints/{}.ckpt".format(current_model_name))
                    consecutive_validation = 0
                else:
                    consecutive_validation += 1

            if consecutive_validation >= 25:
            	break
            
        if consecutive_validation >= 25:
            #print("Early stopping")
            break

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, "./checkpoints/{}.ckpt".format(current_model_name)) 
    
    avg_accuracy, avg_loss, pred = sess.run([accuracy, loss, predictions], feed_dict={
                                            tensor_X: X_val,
                                            tensor_Y: Y_val,
                                            keep_prob: 1.0 })

f = open("results.txt","a+")
acc = round(accuracy_score(Y_val, pred), 3)
prec = round(precision_score(Y_val, pred), 3)
rec = round(recall_score(Y_val, pred), 3)
auc = round(roc_auc_score(Y_val, pred), 3)
f.write(current_model_name + " " + str(acc) + " " + str(prec) + " " + str(rec) + " " + str(auc) + "\n")
f.close()