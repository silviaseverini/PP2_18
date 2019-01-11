import argparse
import tensorflow as tf
import network
import pickle
import processing


def write_not_predicted_values(filename, sequence):

	for i in sequence:
		filename.write(i + "\t" + "NA" + "\t" + "NA\n")


parser = argparse.ArgumentParser(description='Generate your predictions')
parser.add_argument('-i','--input', help='Input file path', required=True)
parser.add_argument('-o','--output', help='Output file path', required=True)
args = vars(parser.parse_args())

# Get lines in input file
input_file = [line.rstrip('\n') for line in open(args["input"])]

# Divide headers and sequences
headers = [input_file[i] for i in range(len(input_file)) if i % 2 == 0]
sequences = [input_file[i] for i in range(len(input_file)) if i % 2 != 0]

# Load vocab locally
with open('./dumps/vocab.pickle', 'rb') as handle:
    vocab = pickle.load(handle)

# Load parameters
with open('./dumps/parameters.pickle', 'rb') as handle:
    parameters = pickle.load(handle)

# TF variables
tf.reset_default_graph()

# Placeholders
tensor_X = tf.placeholder(tf.int32, (None, parameters["timestamps"]), 'inputs')
tensor_Y = tf.placeholder(tf.int32, (None), 'output')
keep_prob = tf.placeholder(tf.float32, (None), 'dropout_input')

# Create graph for the network
logits = network.create_network(tensor_X, tensor_Y, keep_prob, vocab, parameters["embedding_size"])
scores = tf.nn.softmax(logits)
predictions = tf.to_int32(tf.argmax(scores, axis=1))

# TF inference
saver = tf.train.Saver()
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	saver.restore(sess, "./checkpoints/model.ckpt") 

	print("Model restored.")

	# Create output file
	file = open(args["output"],"w") 
 
	for i in range(len(sequences)):
		# Write header
		file.write(headers[i] + "\n") 
		
		# Create ngrams from sequence for inputs
		X, _ = processing.create_input([sequences[i]], None, parameters["sliding_window_size"], parameters["n_grams"])

		# Find residues that are not possible to be predicted (because of sliding window approach) => Solution could be padding
		tmp = int(0.5 * parameters["sliding_window_size"] - 0.5)

		# Print on file extremes of sequence 
		write_not_predicted_values(file, sequences[i][:tmp])
		k = 0
		for j in range(len(X)):
			central_gram_index = len(X[j]) // 2
			central_residue_index = len(X[j][central_gram_index]) // 2

			# Debug
			assert(sequences[i][tmp+k] == X[j][central_gram_index][central_residue_index])
			k+=1

			# Perform inference
			certainty, pred = sess.run([scores, predictions], feed_dict={tensor_X: [vocab.text_to_indices(X[j])], keep_prob: 1.0 })
			file.write(X[j][central_gram_index][central_residue_index] + "\t" + 
				str('+' if pred[0] == 1 else '-') + "\t" + 
				str(round(max(certainty[0]),3)) + "\n") 

		write_not_predicted_values(file, sequences[i][-tmp:])

	file.close() 

	print("End prediction.")
	print("!!!! WARNING 1 !!!: In the output file, instead of percentage > 50 is class 1 and < 50 is class 0, I have a softmax as last layer, so I display the certainty that this is the class (100% means that the model is sure this is the correct class)")
	print("!!! WARNING 2 !!!: Some predictions will be NA, because at the extremes of the sequence, I cannot predict those residues, because of the sliding window approach")