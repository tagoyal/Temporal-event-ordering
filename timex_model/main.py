import os
import random
import numpy as np
import json
from utils_timex import *
from model import *
import string
import pickle as pk
import h5py



output_folder = "./model_timex_pairs_elmo/"
timex_pairs_file = "timex_pairs.txt"
timex_words = pk.load(open("word_timex.txt", "rb"))


dictionary = {"BEFORE": 0, "AFTER": 1, "SIMULTANEOUS": 2, "VAGUE": 3, "INCLUDES": 4, "IS_INCLUDED": 5}
comp_label = {0:1, 1:0, 2:2, 3:3, 4:5, 5:4}






"""


f = open(timex_pairs_file)
timex_pair = []
i = 0
j = 0
for line in f.readlines():
	line_segments = line.strip().split("\t")
	rel = dictionary[line_segments[2]]
	timex_pair.append(timex_elmo(elmo_embed[j], elmo_embed[j+1], rel))
	i += 1
	j += 2
	if i >= 10000:
		break

i = 0
correct_order = {0:1, 1:0}
for d in timex_words:
	timex_pair.append(timex_elmo(elmo_embed[j], elmo_embed[j+1], correct_order[d[2]]))
	j += 2

random.shuffle(timex_pair)
training_data = timex_pair[0:int(0.8*len(timex_pair))]
test_data = timex_pair[int(0.8*len(timex_pair)):len(timex_pair)]



build_elmo_model_seq(training_data, test_data, output_folder)

exit()
"""
training_data_size = 50000
test_data_size = 1000
"""
training_data = [create_pairs_older() for i in range(training_data_size)]
test_data = [create_pairs_older() for i in range(test_data_size)]


f_out= open("timex_pairs_elmo_text.txt", "w")
for d in training_data + test_data:
	f_out.write(d[0][0] + "\n")
	f_out.write(d[1][0] + "\n")

f_out.close()

f_out_label = open("timex_pair_elmo_label", "w")
for d in training_data + test_data:
	f_out_label.write(str(d[2]) + "\n")

f_out_label.close()

exit()
"""

training_data = []
test_data = []

labels = []
f = open("timex_pair_elmo_label")
for line in f.readlines():
	labels.append(int(line.strip()))

filename = 'timex_pairs_elmo_text.hdf5'
data = h5py.File(filename, 'r')

data_keys = sorted([int(i) for i in data.keys()])
elmo_embed = []
for d in data_keys:
	elmo_embed.append(data[str(d)].value)

class timex_elmo(object):
	def __init__(self, elmo1, elmo2, label):
		
		self.elmo1 = np.concatenate((elmo1[0], elmo1[1], elmo1[2]), axis = 1)
		self.elmo2 = np.concatenate((elmo2[0], elmo2[1], elmo2[2]), axis = 1)
		self.label = label
		self.elmo1 = np.mean(self.elmo1, axis = 0)
		self.elmo2 = np.mean(self.elmo2, axis = 0)

all_data = []
for i in range(len(labels)):
	all_data.append(timex_elmo(elmo_embed[2*i], elmo_embed[2*i + 1], labels[i]))

training_data = all_data[:training_data_size]
test_data = all_data[training_data_size: ]

random.shuffle(training_data)
random.shuffle(test_data)

training_data = training_data[:2000]
"""
training_data = process_data(training_data)
test_data = process_data(test_data)

words = []
for d in training_data + test_data:
	words += d.timex1_tokens + d.timex2_tokens


words = list(set(words))
word_vectors_path = "/scratch/cluster/tanya/glove/glove.840B.300d.txt"
word_vectors = read_word_embeddings(word_vectors_path, 300, words)


vocab = list(string.ascii_lowercase) + [str(i) for i in range(0,10)] 
char_embedding = generate_embedding(vocab)

if os.path.exists(output_folder) == False:
	os.makedirs(output_folder)

with open(os.path.join(output_folder, 'char_embedding.pkl'), 'wb') as output:
	pk.dump(char_embedding, output, protocol =2)
"""
print("here")
build_elmo_model(training_data, test_data)


exit()


f = open(timex_pairs_file)
f1 = open("elmo_timex_pairs_10k.txt", "w")
i = 0
for line in f.readlines():
	line_segments = line.strip().split("\t")
	t1 = line_segments[0]
	t2 = line_segments[1]
	f1.write(t1)
	f1.write("\n")
	f1.write(t2)
	f1.write("\n")
	i += 1
	if i >= 10000:
		break

more_data = [create_pairs() for i in range(5000)]
for d in more_data:
	f1.write(d[0])
	f1.write("\n")
	f1.write(d[1])
	f1.write("\n")

f1.close()

with open("word_timex.txt", 'wb') as output:
	pk.dump(more_data,output)



exit()
