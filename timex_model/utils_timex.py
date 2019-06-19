import numpy as np
import random
import tensorflow as tf
from utils import *
import pickle as pk
import re
import string


def read_word_embeddings(embeddings_file, dim, words_to_keep):
    f = open(embeddings_file)
    word_indexer = Indexer()
    vectors = []
    word_indexer.get_index("UNK")
    vectors.append(np.zeros(dim))
    for line in f:
        if line.strip() != "":
            space_idx = line.find(' ')
            word = line[:space_idx]
            if word not in words_to_keep:
            	continue
            else:
            	words_to_keep.remove(word)
            numbers = line[space_idx+1:]
            float_numbers = [float(number_str) for number_str in numbers.split()]
            #print repr(float_numbers)
            vector = np.array(float_numbers)
            word_indexer.get_index(word)
            vectors.append(vector)
            #print repr(word) + " : " + repr(vector)
    f.close() 
    # Add an UNK token at the end
    
    # Turn vectors into a 2-D numpy array
    print("Read in " + repr(len(word_indexer)) + " vectors of size " + repr(vectors[0].shape[0]))
    return WordEmbeddings(word_indexer, np.array(vectors))

class WordEmbeddings(object):
    def __init__(self, word_indexer, vectors):
        self.word_indexer = word_indexer
        self.vectors = vectors

    def get_embedding(self, word):
        word_idx = self.word_indexer.get_index(word)
        if word_idx != -1:
            return self.vectors[word_idx]
        else:
            return self.vectors[word_indexer.get_index("UNK")]


Months_ordering = [["January","Jan"], ["February", "Feb"], ["March", "Mar"], ["April", "Apr"], ["May"], ["June", "Jun"], 
				["July", "Jul"], ["August", "Aug"], ["September", "Sep"], ["October", "Oct"], ["November", "Nov"], ["December", "Dec"]]
Days = [["monday", "mon"], ["tuesday", "tue"], ["wednesday", "wed"], ["thursday", "thurs"], ["friday", "fri"], ["saturday", "sat"], ["sunday", "sun"]]

year_range_start = 1850
year_range_end = 2000
month_range_start = 1
month_range_end = 12
day_range_start = 1
day_range_end = 31
day_suffix = ["", "st", "nd", "rd", "th"]
days_of_w_start = 1
days_of_w_end = 7

def date_generation_mm_dd_yyyy():
	year = str(np.random.randint(year_range_start, year_range_end + 1))
	month = str(np.random.randint(month_range_start, month_range_end + 1))
	day = str(np.random.randint(day_range_start, day_range_end + 1))
	if len(month) == 1:
		month = "0" + month
	if len(day) == 1:
		day = "0" + day
	prob = random.uniform(0,1)
	if prob > 0.5:
		year_str = str(year)
	else:
		year_str = str(year)[2:5]
	timex = month + " " + day + " " + year_str
	return [timex, int(year), int(month), int(day)]


def month_generation():
	month_idx = np.random.randint(month_range_start, month_range_end + 1)
	month_options = Months_ordering[month_idx - 1]
	month = month_options[np.random.randint(0, len(month_options))]
	return [month, None, month_idx, None]

def day_generation():
	day_idx = np.random.randint(days_of_w_start, days_of_w_end + 1)
	day_options = Days[day_idx - 1]
	day = day_options[np.random.randint(0, len(day_options))]
	return [day, day_idx]

def year_generation():
	year = np.random.randint(year_range_start, year_range_end + 1)
	prob = random.uniform(0,1)
	if prob > 0.5:
		year_str = str(year) + "s"
	else:
		year_str = str(year)
	return [year_str, year, None, None]

def date_generation_dd_mmm_yy():
	year = np.random.randint(year_range_start, year_range_end + 1)
	month_idx = np.random.randint(month_range_start, month_range_end + 1)
	month_options = Months_ordering[month_idx - 1]
	month = month_options[np.random.randint(0, len(month_options))]
	day = np.random.randint(day_range_start, day_range_end + 1)
	prob = random.uniform(0,1)
	if prob > 0.5:
		year_str = str(year)
	else:
		#year_str = str(year)[2:5]
		year_str = str(year)

	day_str = str(day) #+ day_suffix[np.random.randint(0, len(day_suffix))]
	
	timex = month + " " + day_str + " " + year_str
	return [timex, year, month_idx, day]

def date_generation_mm_yy():
	year = np.random.randint(year_range_start, year_range_end + 1)
	month_idx = np.random.randint(month_range_start, month_range_end + 1)
	month_options = Months_ordering[month_idx - 1]
	month = month_options[np.random.randint(0, len(month_options))]
	prob = random.uniform(0,1)
	if prob > 0.5:
		year_str = str(year)
	else:
		#year_str = str(year)[2:5]
		year_str = str(year)

	timex = month + " " + year_str
	return [timex, year, month_idx, None]

numbers = ["one", "two", "three" , "four", "five", "six", "seven", "eight", "nine"]
time_ranges = ["day", "week", "month", "year"]
identifier = ["past", "ago", "earlier", "next", "coming"]
extra = ["tomorrow", "yesterday", "now"]

def word_timex_generation():
	prob = random.uniform(0,1)
	order = -1
	while order == -1:
		if prob < 0.2:
			n1 = np.random.randint(0, len(numbers))
			n2 = np.random.randint(0, len(numbers))
			time_range = np.random.randint(0, len(time_ranges))
			ident = random.choice([identifier[1], identifier[2]])
			timex1 = numbers[n1] + " " + time_ranges[time_range] + " " + ident
			timex2 = numbers[n2] + " " + time_ranges[time_range] + " " + ident
			if n1 < n2:
				order = 1
			elif n2 < n1:
				order = 0
			else:
				order = -1
		elif prob < 0.3:
			n1 = np.random.randint(0, len(numbers))
			n2 = np.random.randint(0, len(numbers))
			time_range = np.random.randint(0, len(time_ranges))
			ident = identifier[0]
			timex1 = ident + " " + numbers[n1] + " " + time_ranges[time_range]
			timex2 = ident + " " + numbers[n2] + " " + time_ranges[time_range]
			if n1 < n2:
				order = 1
			elif n2 < n1:
				order = 0
			else:
				order = -1
		elif prob < 0.5:
			n1 = np.random.randint(0, len(numbers))
			n2 = np.random.randint(0, len(numbers))
			time_range = np.random.randint(0, len(time_ranges))
			ident = random.choice([identifier[3], identifier[4]])
			timex1 = ident + " " + numbers[n1] + " " + time_ranges[time_range] 
			timex2 = ident + " " + numbers[n2] + " " + time_ranges[time_range]
			if n1 < n2:
				order = 0
			elif n2 < n1:
				order = 1
			else:
				order = -1
		elif prob < 0.6:
			n1 = np.random.randint(0, len(numbers))
			time_range = np.random.randint(0, len(time_ranges))
			ident = random.choice([identifier[1], identifier[2], identifier[0]])
			timex1 = numbers[n1] + " " + time_ranges[time_range] + " " + ident
			timex2 = "now"
			order = 0
		elif prob < 0.7:
			n1 = np.random.randint(0, len(numbers))
			time_range = np.random.randint(0, len(time_ranges))
			ident = random.choice([identifier[3], identifier[4]])
			timex1 = ident + " " + numbers[n1] + " " + time_ranges[time_range]
			timex2 = "now"
			order = 1
		elif prob < 0.73:
			timex1 = "tomorrow"
			timex2 = "now"
			order = 1
		elif prob < 0.76:
			timex1 = "yesterday"
			timex2 = "now"
			order = 0
		elif prob < 0.8:
			timex1 = "yesterday"
			timex2 = "tomorrow"
			order = 0
		elif prob < 1:
			n1 = np.random.randint(0, len(numbers))
			n2 = np.random.randint(0, len(numbers))
			time_range = np.random.randint(0, len(time_ranges))
			ident1 = random.choice([identifier[1], identifier[2], identifier[0]])
			ident2 = random.choice([identifier[3], identifier[4]])
			timex1 = numbers[n1] + " " + time_ranges[time_range] + " " + ident1
			timex2 = ident2 + " " + numbers[n2] + " " + time_ranges[time_range] 
			order = 0

	prob = random.uniform(0,1)
	if prob < 0.5:
		t1 = timex1
		t2 = timex2
		r = order
	else:
		t1 = timex2
		t2 = timex1
		r = 0 if order == 1 else 1

	return [t1, t2, r]

def compare_timex(timex1, timex2):
	if timex1[1] == None and timex2[1] != None:
		return -1
	elif timex1[1] != None and timex2[1] == None:
		return -1
	elif timex1[1] == None and timex2[1] == None:
		if timex1[2] < timex2[2]:
			return 0
		elif timex2[2] < timex1[2]:
			return 1
		else:
			return -1
	elif timex1[2] == None or timex2[2] == None:
		if timex1[1] < timex2[1]:
			return 0
		elif timex2[1] < timex1[1]:
			return 1
		else:
			return -1
	elif timex1[3] == None or timex2[3] == None:
		if timex1[1] < timex2[1]:
			return 0
		elif timex2[1] < timex1[1]:
			return 1
		elif timex1[2] < timex2[2]:
			return 0
		elif timex2[2] < timex1[2]:
			return 1
		else:
			return -1
	else:
		if timex1[1] < timex2[1]:
			return 0
		elif timex2[1] < timex1[1]:
			return 1
		elif timex1[2] < timex2[2]:
			return 0
		elif timex2[2] < timex1[2]:
			return 1
		if timex1[3] < timex2[3]:
			return 0
		elif timex2[3] < timex1[3]:
			return 1
		else:
			return -1


def create_pairs():
	prob = random.uniform(0,1)
	order = -1
	while order == -1:
		if prob < 0.15:
			timex1 = date_generation_dd_mmm_yy()
			timex2 = date_generation_dd_mmm_yy()
			prob1 = random.uniform(0,1)
			if prob1 < 0.3:
				timex2[1] = timex1[1]
				prob2 = random.uniform(0,1)
				if prob2 < 0.3:
					timex2[2] = timex1[2]
			order = compare_timex(timex1, timex2)
		elif prob < 0.3:
			timex1 = date_generation_mm_yy()
			timex2 = date_generation_mm_yy()
			prob1 = random.uniform(0,1)
			if prob1 < 0.3:
				timex2[1] = timex1[1]
			order = compare_timex(timex1, timex2)
		elif prob < 0.35:
			timex1 = month_generation()
			timex2 = month_generation()
			order = compare_timex(timex1, timex2)
		elif prob < 0.5:
			timex1 = year_generation()
			timex2 = year_generation()
			order = compare_timex(timex1, timex2)
		elif prob < 0.65:
			timex1 = date_generation_mm_yy()
			timex2 = year_generation()
			order = compare_timex(timex1, timex2)
		elif prob < .75:
			timex1 = date_generation_dd_mmm_yy()
			timex2 = year_generation()
			order = compare_timex(timex1, timex2)
		elif prob < .85:
			timex1 = date_generation_dd_mmm_yy()
			timex2 = date_generation_mm_yy()
			prob1 = random.uniform(0,1)
			if prob1 < 0.3:
				timex2[1] = timex1[1]
			order = compare_timex(timex1, timex2)
		elif prob < 1:
			timex1, timex2, order = word_timex_generation()
			timex1 = [timex1, None, None, None]
			timex2 = [timex2, None, None, None]

	return [timex1, timex2, order]


class timex_pair:
	def __init__(self, timex1, timex2, label):
		exclude = set(string.punctuation)
		timex1 = ''.join(str(ch) for ch in timex1 if str(ch) not in exclude)
		timex2 = ''.join(str(ch) for ch in timex2 if str(ch) not in exclude)
		self.timex1_chars = [c.lower() for c in list(timex1)]
		self.timex2_chars = [c.lower() for c in list(timex2)]
		self.timex1_tokens = [c.lower() for c in timex1.split()]
		self.timex2_tokens = [c.lower() for c in timex2.split()]
		self.timex1 = timex1
		self.timex2 = timex2
		self.label = label

def process_data(data):
	data_processed = []
	for d in data:
		data_processed.append(timex_pair(d[0][0], d[1][0], d[2]))
	return data_processed

def generate_embedding(vocab):
	vocab_indexer = Indexer()
	vector_size = len(vocab)
	vectors = []
	for i,v in enumerate(vocab):
		vocab_indexer.get_index(v)
		vector = np.zeros(vector_size)
		vector[i] = 1
		vectors.append(vector)

	vocab_indexer.get_index(" ")
	vector = np.zeros(vector_size)
	vectors.append(vector)

	return WordEmbeddings(vocab_indexer, np.array(vectors))
    

def pad_to_length(np_arr, length):
    result = np.zeros(length)
    result[0:np_arr.shape[0]] = np_arr
    return result

def pad_redundant_rows(vec, row_size):
	result = np.zeros((row_size, len(vec)))
	result[0] = vec
	return result	

class Timex_Model(object):
	seq_x = None
	seq_len_x = None
	sum_x = None
	sess = None
	saver = None
	graph = None
	char_embeddings = None


	def __init__(self, folder_name):
		self.sess =  tf.Session()
		self.saver = tf.train.import_meta_graph(folder_name + '/model.ckpt.meta')
		self.saver.restore(self.sess, tf.train.latest_checkpoint(folder_name))
		self.graph = tf.get_default_graph()

		self.seq_x = self.graph.get_tensor_by_name('p1:0')
		self.seq_len_x = self.graph.get_tensor_by_name('p3:0')
		self.sum_x = self.graph.get_tensor_by_name('sum_x:0')

		self.char_embeddings = pk.load(open(folder_name + '/char_embedding.pkl', "rb"))

	def get_timex_embed(self, tokens, seq_len_max):
		tokens = re.findall(r"[\w']", tokens)
		tokens =  [t.lower() for t in tokens]
		
		train_timex1 = pad_redundant_rows(pad_to_length(np.array(self.char_embeddings.word_indexer.get_index_list(list(tokens))), 20), 16)
		train_length1 = np.array([len(list(tokens))] + [0] * 15)

		[output] = self.sess.run([ self.sum_x], feed_dict = {self.seq_x: train_timex1, self.seq_len_x: train_length1})
		return output[0]

	
	def close_session(self):
		self.sess.close()
		tf.Session().close()
		print("\n\n\n\ndid reach here")

def date_generation_yyyy():
	year = np.random.randint(year_range_start, year_range_end + 1)
	return [str(year), year, None, None]

def date_generation_yyyy_mm():
	year = np.random.randint(year_range_start, year_range_end + 1)
	month = str(np.random.randint(month_range_start, month_range_end + 1))
	if len(month) == 1:
		month = "0" + month
	timex = str(year) + "-" + str(month)
	return [timex, year, month, None]

def date_generation_yyyy_mm_dd():
	year = np.random.randint(year_range_start, year_range_end + 1)
	month = str(np.random.randint(month_range_start, month_range_end + 1))
	if len(month) == 1:
		month = "0" + month
	day = str(np.random.randint(day_range_start, day_range_end + 1))
	timex = str(year) + "-" + str(month) + "-" + str(day)
	return [timex, year, month, day]

def date_generation_yyyy_Q():
	year = np.random.randint(year_range_start, year_range_end + 1)
	q = np.random.randint(1,5)
	timex = str(year) + "-Q" + str(q)
	return [timex, year, (q-1)*3 + 1, None]


suffixes = ["Y", "D", "M", "W", "H"]
def data_generation_suffix():
	s = np.random.randint(0, 5)
	num = np.random.randint(1, 30)
	if s != 4:
		timex = "P" + str(num) + suffixes[s]
	else:
		timex = "PT" + str(num) + suffixes[s] 

	return [timex, s, num]

def compare_timex_values(timex1, timex2):

	if timex1[1] == None or timex2[1] == None:
		return -1
	elif timex1[1] < timex2[1]:
		return 0
	elif timex2[1] < timex1[1]:
		return 1
	elif timex1[2] == None or timex2[2] == None:
		return -1
	elif timex1[2] < timex2[2]:
		return 0
	elif timex2[2] < timex1[2]:
		return 1
	elif timex1[3] == None or timex2[3] == None:
		return -1
	elif timex1[3] < timex2[3]:
		return 0
	elif timex2[3] < timex1[3]:
		return 1
	else:
		return -1

def create_pairs_values():
	index = np.random.randint(1, 8)
	order = -1
	while order == -1:
		if index == 1:
			timex1 = date_generation_yyyy()
			timex2 = date_generation_yyyy()
			order = compare_timex_values(timex1, timex2)
		if index == 2:
			timex1 = date_generation_yyyy_mm()
			timex2 = date_generation_yyyy_mm()
			prob = random.uniform(0,1)
			if prob < 0.6:
				timex2[1] = timex1[1]
				prob = random.uniform(0,1)
			order = compare_timex_values(timex1, timex2)
		if index == 3:
			timex1 = date_generation_yyyy_mm_dd()
			timex2 = date_generation_yyyy_mm_dd()
			prob = random.uniform(0,1)
			if prob < 0.6:
				timex2[1] = timex1[1]
				prob = random.uniform(0,1)
				if prob < 0.4:
					timex2[2] = timex1[2]
			order = compare_timex_values(timex1, timex2)
		if index == 4:
			timex1 = date_generation_yyyy_Q()
			timex2 = date_generation_yyyy_Q()
			prob = random.uniform(0,1)
			if prob < 0.6:
				timex2[1] = timex1[1]
			order = compare_timex_values(timex1, timex2)
		if index == 5:
			timex1 = date_generation_yyyy()
			timex2 = date_generation_yyyy_mm()
			order = compare_timex_values(timex1, timex2)
		if index == 6:
			timex1 = date_generation_yyyy()
			timex2 = date_generation_yyyy_mm_dd()
			order = compare_timex_values(timex1, timex2)
		if index == 7:
			timex1 = date_generation_yyyy_mm()
			timex2 = date_generation_yyyy_mm_dd()
			prob = random.uniform(0,1)
			if prob < 0.4:
				timex2[1] = timex1[1]
			order = compare_timex_values(timex1, timex2)
	return [timex1, timex2, order]

