from os import listdir
import json, random
from utils import *
from utils_temporal import *
from models import *
import pickle as pk
import h5py


word_vectors_path = "./data/glove.6B.50d.txt"
input_folder_data = "./data/distant_data/"



if __name__ == '__main__':
	word_vectors = read_word_embeddings(word_vectors_path, 50)

	tag_indexer = pk.load(open(os.path.join(input_folder_data, 'tag_indexer.pkl'), "rb"))
	
	train_data = pk.load(open(os.path.join(input_folder_data, 'train_data.pkl'), "rb"))	
	test_data = pk.load(open(os.path.join(input_folder_data, 'test_data.pkl'), "rb"))

	k = 3000
	build_model(train_data[0:k], train_data[k:k+400], word_vectors, tag_indexer, output_folder)
	