

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

dictionary = {"BEFORE": 0, "AFTER": 1, "SIMULTANEOUS": 2, "VAGUE": 3, "INCLUDES": 4, "IS_INCLUDED": 5}
comp_label = {0:1, 1:0, 2:2, 3:3, 4:5, 5:4}

training_data_size = 50000
test_data_size = 1000

training_data = [create_pairs() for i in range(training_data_size)]
test_data = [create_pairs() for i in range(test_data_size)]

training_data = process_data(training_data)
test_data = process_data(test_data)

random.shuffle(training_data)
random.shuffle(test_data)

vocab = list(string.ascii_lowercase) + [str(i) for i in range(0,10)] 
char_embedding = generate_embedding(vocab)

if os.path.exists(output_folder) == False:
	os.makedirs(output_folder)

with open(os.path.join(output_folder, 'char_embedding.pkl'), 'wb') as output:
	pk.dump(char_embedding, output, protocol =2)

build_model(training_data, test_data)
