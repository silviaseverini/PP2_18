import numpy as np
import pickle
import src.dictionary as dictionary
import src.utils as utils


# Prepare dataset for prediction
protein_names, sequences, labels = [], [], []
    
# Open file containing dataset    
with open('./dataset/ppi_data.fasta') as f:
    lines = f.read().splitlines()
    
    for i in range(len(lines)):
        
        if i % 3 == 0:
            protein_names.append(lines[i])
        elif i % 3 == 1:
            sequences.append(list(lines[i]))
        elif i % 3 == 2:
            labels.append(np.array([utils.convert_label(letter) for letter in lines[i]]))
            
protein_names = np.array(protein_names)
sequences = np.array(sequences)
labels = np.array(labels)

assert(protein_names.shape[0] == sequences.shape[0] == labels.shape[0])

print(protein_names[0])
print("".join(sequences[0]))
print(labels[0].shape) 

# Split percentage of training and validation
split_percentage = 0.8

# Count how many samples into training dataset
total_dataset = len(sequences)
train_dataset = int(total_dataset * split_percentage)

# Shuffle
indices = list(range(total_dataset))
np.random.shuffle(indices)

# Train dataset
sequences_train = sequences[indices[:train_dataset]]
labels_train = labels[indices[:train_dataset]]

# Validation dataset
sequences_val = sequences[indices[train_dataset:]]
labels_val = labels[indices[train_dataset:]]

# Shapes
print("Training samples: " + str(sequences_train.shape[0]))
print("Validation samples: " + str(sequences_val.shape[0]))

# Balance labels because negative are much more than positives
balanced_sequences_train = np.array(sequences_train, copy=True)  
balanced_labels_train = np.array(labels_train, copy=True)
while True:
    
    index = np.random.choice(len(labels_train), 1, replace=True)
    split_pos, split_neg = utils.get_total_pos_neg(labels_train[index])
    
    if split_pos > split_neg:
        balanced_sequences_train = np.append(balanced_sequences_train, sequences_train[index], axis=0)
        balanced_labels_train = np.append(balanced_labels_train, labels_train[index], axis=0)
        
    tot_pos, tot_neg = utils.get_total_pos_neg(balanced_labels_train)
    if tot_pos >= tot_neg:
        break
        
print("Final total Positive and negative labels " + str(utils.get_total_pos_neg(balanced_labels_train)))

# Create vocabulary of n-grams
vocab = dictionary.LanguageDictionary(balanced_sequences_train)
max_length_seq = utils.max_length_sentence(balanced_sequences_train)

X_train, Y_train = utils.create_input(vocab, max_length_seq, balanced_sequences_train, balanced_labels_train)
X_val, Y_val = utils.create_input(vocab, max_length_seq, sequences_val, labels_val)

print(X_train.shape)
print(X_val.shape)

# Save dump
save_dump = { "X_train" : X_train, "Y_train" : Y_train, "X_val" : X_val, "Y_val" : Y_val, "vocab" : vocab }

with open('./dumps/dump.pickle', 'wb') as handle:
    pickle.dump(save_dump, handle, protocol=pickle.HIGHEST_PROTOCOL)