import argparse
import pickle
import tensorflow as tf
import src.network as network
import src.utils as utils

parser = argparse.ArgumentParser(description='Generate your predictions')
parser.add_argument('-i','--input', help='Input file path', required=True)
parser.add_argument('-o','--output', help='Output file path', required=True)
args = vars(parser.parse_args())

# Params
current_model_name = "model-0.0001-0.0001-1.0-64-32-128"

splitted = current_model_name.split("-")
embedding_size = int(splitted[4])
hidden_cells = int(splitted[6])

# Get lines in input file
input_file = [line.rstrip('\n') for line in open(args["input"])]

# Divide headers and sequences
headers = [input_file[i] for i in range(len(input_file)) if i % 2 == 0]
sequences = [input_file[i] for i in range(len(input_file)) if i % 2 != 0]
assert(len(headers) == len(sequences))

# Load parameters
with open('./dumps/dump.pickle', 'rb') as handle:
    dumped = pickle.load(handle)
vocab = dumped["vocab"]
max_length_seq = dumped["X_train"].shape[1]

# Placeholders
tensor_X = tf.placeholder(tf.int32, (None, max_length_seq), 'inputs')
tensor_Y = tf.placeholder(tf.int32, (None, None), 'outputs')
keep_prob = tf.placeholder(tf.float32, (None), 'dropout_keep')

# Network
logits, mask, sequence_length = network.create_network(tensor_X, tensor_Y, keep_prob, vocab, embedding_size, hidden_cells)

# Useful tensors
scores = tf.nn.softmax(logits)
predictions = tf.to_int32(tf.argmax(scores, axis=2))

with tf.Session() as sess:
	saver = tf.train.Saver()
	sess.run(tf.global_variables_initializer())
	sess.run(tf.local_variables_initializer())
	saver.restore(sess, "./checkpoints/{}.ckpt".format(current_model_name))

	# Create output file
	file = open(args["output"], "w") 
	
	for i in range(len(headers)):
		# Write header
		file.write(headers[i] + "\n") 

		tmp = utils.pad_sentence(vocab.text_to_indices(list(sequences[i])), max_length_seq)

		certainty, pred = sess.run([scores, predictions], feed_dict={ tensor_X: [tmp], 
												 					  keep_prob: 1.0  })

		for j in range(len(sequences[i])):
			file.write(sequences[i][j] + "\t" + str('+' if pred[0][j] == 1 else '-') + "\t" + str(round(max(certainty[0][j]), 3)) + "\n")

	file.close()


print("!!!! WARNING 1 !!!: In the output file, instead of percentage > 50 is class 1 and < 50 is class 0, I have a softmax as last layer, so I display the certainty that this is the class (100% means that the model is sure this is the correct class)")