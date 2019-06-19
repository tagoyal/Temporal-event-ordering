import os, json, csv
import re
from utils import *
import numpy as np
from os import listdir
from random import shuffle
from bs4 import BeautifulSoup
import random
import tensorflow as tf


relations = ["b", "a", "s", "v"]
relations_reverse = {"a": "b", "b" : "a", "s" : "s", "ii": "i", "i" : "ii", "v" : "v"}
relations_transitivity = [ [["b"], relations, ["b"], ["v", "b"]],
							[relations, ["a"], ["a"], [ "v", "a"]],
							[["b"], ["a"], ["s"], ["v"]],
							[["b", "v"], ["v", "a"], ["v"], relations]]
relations_dict = {'BEFORE': "b", 'AFTER': "a", 'SIMULTANEOUS': "s", 'IS_INCLUDED' : "a", 'INCLUDES' : "b"}

class WordEmbeddings:
    def __init__(self, word_indexer, vectors):
        self.word_indexer = word_indexer
        self.vectors = vectors

    def get_embedding(self, word):
        word_idx = self.word_indexer.get_index(word)
        if word_idx != -1:
            return self.vectors[word_idx]
        else:
            return self.vectors[word_indexer.get_index("UNK")]

def read_word_embeddings(embeddings_file, dim):
    f = open(embeddings_file)
    word_indexer = Indexer()
    vectors = []
    word_indexer.get_index("UNK")
    vectors.append(np.zeros(dim))
    for line in f:
        if line.strip() != "":
            space_idx = line.find(' ')
            word = line[:space_idx]
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

class IndexedExample(object):
    def __init__(self, dep_path1, dep_path2, label, sent1, sent2, pos1, pos2, rel1, rel2, eid1, eid2, doc, sent1_id, sent2_id, doc_name, possible_rels, tags1, tags2, tid_parents1, tid_parents2):
    	self.dep_path1 = dep_path1
    	self.dep_path2 = dep_path2
    	self.label = label
    	self.sent1 = [x.lower() for x in sent1]
    	self.sent2 = [x.lower() for x in sent2]
    	self.pos1 = pos1
    	self.pos2 = pos2
    	self.rel1 = rel1
    	self.rel2 = rel2
    	self.eid1 = eid1
    	self.eid2 = eid2
    	self.doc = doc
    	self.sent1_id = sent1_id
    	self.sent2_id = sent2_id
    	self.doc_name = doc_name
    	self.possible_rels = possible_rels
    	self.mask = random.randint(0,1)
    	self.tags1 = tags1
    	self.tags2 = tags2
    	self.e1_idx = dep_path1[0]
    	self.e2_idx = dep_path2[0]
    	self.elmo1 = None
    	self.elmo2 = None
    	self.tid_parents1 = tid_parents1
    	self.tid_parents2 = tid_parents2

def find_tag(tokenized_tags, tlink):
	for i, sent in enumerate(tokenized_tags):
		if tlink in sent:
			return i
	return -1

def get_parent(tid, tokenized_tags, dependencies, tokenized_text):
	idxs = [i for i,x in enumerate(tokenized_tags) if x == tid]
	words = [tokenized_text[i] for i in idxs]

	indices = []
	for w in words:
		i = [dep["word_idx"] for dep in dependencies if dep["word"] == w]
		if len(i) > 0:
			indices.append(i[0])

	p = None
	for i in indices:
		for dep in dependencies:
			if dep["word_idx"] == i:
				if "mod" in dep["rel"] and dep["head_idx"] not in indices:
					p = dep["head_word"]

	try:
		idx1 = tokenized_text.index(p)
	except:
		return -1

	return idx1

def get_dep_path_to_root(tid, tokenized_tags, dependencies, tokenized_text):
	idx = tokenized_tags.index(tid)
	words = [tokenized_text[idx]]
	head = -1
	indices = [dep["word_idx"] for dep in dependencies if dep["word"] == words[-1]]
	if len(indices) == 0:
		return -1

	elif len(indices) == 1:
		idx = indices[0]
	else:
		idx = min(indices, key=lambda x:abs(idx-x))

	
	path_indices = [idx]
	while idx != 0:
		for dep in dependencies:
			if dep["word_idx"] == idx:
				idx = dep["head_idx"]
				path_indices.append(idx)
				continue

	path_words = []
	for p in path_indices:
		if p == 0:
			break
		x = [dep["word"] for dep in dependencies if dep["word_idx"] == p][0]
		path_words.append(x)

	path_indices_actual = []
	for w in path_words:
		try:
			path_indices_actual.append(tokenized_text.index(w))
		except:
			continue

	path_indices_actual.append(1000)
	#print(tokenized_text[path_indices_actual[0]])
	return path_indices_actual

def get_dep_path_betw(tid1, tid2, tokenized_tags, dependencies, tokenized_text):
	idx1 = get_dep_path_to_root(tid1, tokenized_tags, dependencies, tokenized_text)
	idx2 = get_dep_path_to_root(tid2, tokenized_tags, dependencies, tokenized_text)


	if idx1 == -1 or idx2 == -1:
		return (-1, -1)

	if idx1[0] in idx2:
		common_ancestor = idx1[0]
	elif idx2[0] in idx1:
		common_ancestor = idx2[0]
	else:
		for w in idx1:
			if w in idx2:
				common_ancestor = w

	indices1 = []
	for i in idx1:
		indices1.append(i)
		if i == common_ancestor:
			break

	indices2 = []
	for i in idx2:
		indices2.append(i)
		if i == common_ancestor:
			break

	return (indices1, indices2)

def get_indexed_words(words, word_vectors):
	indexer = word_vectors.word_indexer
	indexed_words = []
	for w in words:
		indexed_words.append(indexer.get_index(w) if indexer.contains(w) else indexer.get_index("UNK"))
	return indexed_words

def get_possible_rels(e1, e2, timex_list, tlinks):
	
	def normalize(s, e):
		if s[0] == e:
			return s
		else:
			return (s[1], s[0], relations_reverse[s[2]])


	s1 = None
	s2 = None
	possible_rels = []
	for t in timex_list:
		for tl in tlinks:
			if (t, e1) == (tl[0], tl[1]) or (t, e1) == (tl[1], tl[0]):
				s1 = tl
		for tl in tlinks:
			if (t, e2) == (tl[0], tl[1]) or (t, e2) == (tl[1], tl[0]):
				s2 = tl

		if s1 == None or s2 == None:
			break
		if s1[2] in ["i", "ii"] or s2[2] in ["i", "ii"]:
			break

		s1 = normalize(s1, e1)
		s2 = normalize(s2, e2)
		s2 = (s2[1], s2[0], relations_reverse[s2[2]])

		p_rels = relations_transitivity[relations.index(s1[2])][relations.index(s2[2])]
		possible_rels = possible_rels + p_rels

	if len(possible_rels) == 0:
		possible_rels = relations
	return list(set(possible_rels))



def create_data_new(data_raw, word_vectors, doc_links):
	data = []
	not_counted = 0
	total_tlinks = 0
	doc_id = 0
	for doc in data_raw:
		tokenized_text = doc["tokenized_text"]
		tokenized_tags = doc["tokenized_tags"]
		dependencies = doc["dependencies"]

		for tlink in doc["tlinks"]:
			if "t" in tlink[0] or "t" in tlink[1]:
				continue
			total_tlinks += 1
			sent1 = find_tag(tokenized_tags, tlink[0])
			sent2 = find_tag(tokenized_tags, tlink[1])
			if sent1 == -1 or sent2 == -1:
				continue
			elif sent1 == sent2:
				dep_path1, dep_path2 = get_dep_path_betw(tlink[0], tlink[1], tokenized_tags[sent1], dependencies[sent1], tokenized_text[sent1])
			else:
				dep_path1 = get_dep_path_to_root(tlink[0], tokenized_tags[sent1], dependencies[sent1], tokenized_text[sent1])
				dep_path2 = get_dep_path_to_root(tlink[1], tokenized_tags[sent2], dependencies[sent2], tokenized_text[sent2])

			if dep_path1 == -1 or dep_path2 == -1:
				not_counted += 1
			else:
				pos1 = [d["pos"] for d in dependencies[sent1]]
				pos2 = [d["pos"] for d in dependencies[sent2]]
				rel1 = [d["rel"] for d in dependencies[sent1]]
				rel2 = [d["rel"] for d in dependencies[sent2]]
				
				timex_ids1 = []
				for t in tokenized_tags[sent1]:
					if "t" in t:
						timex_ids1.append(t)
				timex_ids1 = set(timex_ids1)
				tid_parents1 = {}
				for t in timex_ids1:
					idx = tokenized_tags[sent1].index(t)
					path = get_dep_path_to_root(t, tokenized_tags[sent1], dependencies[sent1], tokenized_text[sent1])
					if path != -1:
						tid_parents1[t] = path

				timex_ids2 = []
				for t in tokenized_tags[sent2]:
					if "t" in t:
						timex_ids2.append(t)
				timex_ids2 = set(timex_ids2)
				tid_parents2 = {}
				for t in timex_ids2:
					idx = tokenized_tags[sent2].index(t)
					path = get_dep_path_to_root(t, tokenized_tags[sent2], dependencies[sent2], tokenized_text[sent2])
					if path != -1:
						tid_parents2[t] = path

				data.append(IndexedExample(dep_path1, dep_path2, tlink[2], tokenized_text[sent1], tokenized_text[sent2], pos1, pos2, rel1, rel2, tlink[0], tlink[1], doc, sent1, sent2, doc["doc"], None, tokenized_tags[sent1], tokenized_tags[sent2]
					,tid_parents1, tid_parents2))
		doc_id += 1
	return data

def create_data(data_raw, word_vectors):

	data = []
	not_counted = 0
	total_tlinks = 0
	doc_id = 0
	for doc in data_raw:
		tokenized_text = doc["tokenized_text"]
		tokenized_tags = doc["tokenized_tags"]
		dependencies = doc["dependencies"]
		#for tlink in doc["tlinks"]:
		#for tlink in docs_tlinks[doc["doc"]]:
		for tlink in doc["tlinks"]:
			
			if "t" in tlink[0] or "t" in tlink[1]:
				continue
			
			total_tlinks += 1
			sent1 = find_tag(tokenized_tags, tlink[0])
			sent2 = find_tag(tokenized_tags, tlink[1])

			if sent1 == -1 or sent2 == -1:
				continue
			elif sent1 == sent2:
				dep_path1, dep_path2 = get_dep_path_betw(tlink[0], tlink[1], tokenized_tags[sent1], dependencies[sent1], tokenized_text[sent1])
			else:
				dep_path1 = get_dep_path_to_root(tlink[0], tokenized_tags[sent1], dependencies[sent1], tokenized_text[sent1])
				dep_path2 = get_dep_path_to_root(tlink[1], tokenized_tags[sent2], dependencies[sent2], tokenized_text[sent2])

			if dep_path1 == -1 or dep_path2 == -1:
				not_counted += 1
			else:
				pos1 = [d["pos"] for d in dependencies[sent1]]
				pos2 = [d["pos"] for d in dependencies[sent2]]
				rel1 = [d["rel"] for d in dependencies[sent1]]
				rel2 = [d["rel"] for d in dependencies[sent2]]
				
				if dep_path1[-1] == 1000:
					dep_path1 = dep_path1[0:-1]
				if dep_path2[-1] == 1000:
					dep_path2 = dep_path2[0:-1]
				
				timex_ids1 = []
				for t in tokenized_tags[sent1]:
					if "t" in t:
						timex_ids1.append(t)
				timex_ids1 = set(timex_ids1)
				tid_parents1 = {}
				for t in timex_ids1:
					path = get_parent(t, tokenized_tags[sent1], dependencies[sent1], tokenized_text[sent1])
					if path != -1:
						tid_parents1[t] = path

				timex_ids2 = []
				for t in tokenized_tags[sent2]:
					if "t" in t:
						timex_ids2.append(t)
				timex_ids2 = set(timex_ids2)
				tid_parents2 = {}
				for t in timex_ids2:
					path = get_parent(t, tokenized_tags[sent2], dependencies[sent2], tokenized_text[sent2])
					if path != -1:
						tid_parents2[t] = path
				"""
				e1_idx = dep_path1[0]
				e2_idx = dep_path2[0]

				one = False
				two = False
				for t in tid_parents1.keys():
					if tid_parents1[t] == e1_idx:
						one = True
				for t in tid_parents2.keys():
					if tid_parents2[t] == e2_idx:
						two = True

				if one == True and two == True:
					print(tokenized_text[sent1])
					print(tokenized_text[sent2])
					print("\n\n")
				"""
				data.append(IndexedExample(dep_path1, dep_path2, tlink[2], tokenized_text[sent1], tokenized_text[sent2], pos1, pos2, rel1, rel2, tlink[0], tlink[1], doc, sent1, sent2, doc["doc"], relations, tokenized_tags[sent1], tokenized_tags[sent2]
					,tid_parents1, tid_parents2))
		doc_id += 1
	return data

def create_data_unsup(data_raw):
	data = []
	not_counted = 0
	total_tlinks = 0
	doc_id = 0
	for doc in data_raw:
		tokenized_text = doc["tokenized_text"]
		tokenized_tags = doc["tokenized_tags"]
		dependencies = doc["dependencies"]


		for tlink in doc["tlinks"]:
			if "t" in tlink[0] or "t" in tlink[1]:
				continue
			total_tlinks += 1
			sent1 = find_tag(tokenized_tags, tlink[0])
			sent2 = find_tag(tokenized_tags, tlink[1])

			if sent1 == -1 or sent2 == -1:
				continue
			elif sent1 == sent2:

				dep_path1, dep_path2 = get_dep_path_betw(tlink[0], tlink[1], tokenized_tags[sent1], dependencies[sent1], tokenized_text[sent1])
			else:
				dep_path1 = get_dep_path_to_root(tlink[0], tokenized_tags[sent1], dependencies[sent1], tokenized_text[sent1])
				dep_path2 = get_dep_path_to_root(tlink[1], tokenized_tags[sent2], dependencies[sent2], tokenized_text[sent2])

			if dep_path1 == -1 or dep_path2 == -1:
				not_counted += 1
			else:
				pos1 = [d["pos"] for d in dependencies[sent1]]
				pos2 = [d["pos"] for d in dependencies[sent2]]
				rel1 = [d["rel"] for d in dependencies[sent1]]
				rel2 = [d["rel"] for d in dependencies[sent2]]

				if tlink[2] in ['IS_INCLUDED', 'INCLUDES']:
					continue
					
				rel = relations_dict[tlink[2]]

				if dep_path1[-1] == 1000:
					dep_path1 = dep_path1[0:-1]
				if dep_path2[-1] == 1000:
					dep_path2 = dep_path2[0:-1]

				timex_ids1 = []
				for t in tokenized_tags[sent1]:
					if "t" in t:
						timex_ids1.append(t)
				timex_ids1 = set(timex_ids1)
				tid_parents1 = {}
				for t in timex_ids1:
					path = get_parent(t, tokenized_tags[sent1], dependencies[sent1], tokenized_text[sent1])
					if path != -1:
						tid_parents1[t] = path

				timex_ids2 = []
				for t in tokenized_tags[sent2]:
					if "t" in t:
						timex_ids2.append(t)
				timex_ids2 = set(timex_ids2)
				tid_parents2 = {}
				for t in timex_ids2:
					print(t)
					path = get_parent(t, tokenized_tags[sent2], dependencies[sent2], tokenized_text[sent2])
					if path != -1:
						tid_parents2[t] = path
						
				data.append(IndexedExample(dep_path1, dep_path2, tlink[2], tokenized_text[sent1], tokenized_text[sent2], pos1, pos2, rel1, rel2, tlink[0], tlink[1], doc, sent1, sent2, None, None, tokenized_tags[sent1], tokenized_tags[sent2]
					,tid_parents1, tid_parents2))

		doc_id += 1
	return data

def sample(data, count):
	return list(np.random.choice(data,count))

def balance_labels(data):
	label_data = {}
	for d in data:
		if d.label not in label_data.keys():
			label_data[d.label] = []
		label_data[d.label].append(d)

	label_counts = [len(label_data[x]) for x in label_data.keys()]
	max_count = np.max(label_counts)

	data_balanced = []
	for l in label_data.keys():
		data_balanced = data_balanced + sample(label_data[l], max_count)

	shuffle(data_balanced)
	return data_balanced

def create_adjacency_matrix(document):
	eids = [d.eid1 for d in document]
	eids = eids + [d.eid2 for d in document]
	eids = set(eids)
	eid_dict = {e:i for i,e in enumerate(eids)}

	mat = np.zeros((len(eids), len(eids)))
	mat_keep = np.zeros((len(eids), len(eids)))

	for d in document:
		if d.label == "b":
			mat[eid_dict[d.eid1]][eid_dict[d.eid2]] = 1
		elif d.label == "a":
			mat[eid_dict[d.eid2]][eid_dict[d.eid1]] = 1
		else:
			mat_keep[eid_dict[d.eid1]][eid_dict[d.eid2]] = 1

	return mat, eid_dict, mat_keep

def get_links_to_keep(mat):
	mats = []
	mat_copy = mat
	for i in range(mat.shape[0]):
		mat_copy = np.matmul(mat_copy, mat)
		mats.append(mat_copy)

	m_add = np.zeros((mat.shape[0], mat.shape[0]))
	for m in mats:
		m_add = np.add(m_add, m)

	m_add = np.sign(m_add)
	m_add = (m_add - 1)*(-1)

	m_add = m_add + mat 
	m_add[m_add < 1.5] = 0

	return np.sign(m_add)

def get_reduced_data(mat, data_doc, eid_dict):

	nonzero = np.nonzero(mat)
	nonzero_row = nonzero[0]
	nonzero_col = nonzero[1]
	data_final = []
	eid_dict_rev = {i:e for e,i in eid_dict.items()}

	for i,j in zip(nonzero_row, nonzero_col):
		for d in data_doc:
			if d.eid1 == eid_dict_rev[i] and d.eid2 == eid_dict_rev[j]:
				data_final.append(d)
				break
			elif d.eid1 == eid_dict_rev[j] and d.eid2 == eid_dict_rev[i]:
				data_final.append(d)
				break

	return data_final

def remove_redundant_links(data):
	data_links_removed = []
	data_doc_wise = {}
	for d in data:
		if d.doc not in data_doc_wise.keys():
			data_doc_wise[d.doc] = []
		data_doc_wise[d.doc].append(d)

	for d in data_doc_wise.keys():
		mat, eid_dict, mat_keep = create_adjacency_matrix(data_doc_wise[d])
		mat_keep_2 = get_links_to_keep(mat)

		mat_removed = mat - mat_keep_2 - mat_keep

		m = mat_keep_2 + mat_keep

		x = get_reduced_data(m, data_doc_wise[d], eid_dict)
		data_links_removed = data_links_removed + x

	return data_links_removed

def get_tlinks(tlinks_file):
	docs_tlinks = {}
	with open(tlinks_file) as f:
		for line in f.readlines():
			x = line.strip().split("\t")
			doc = x[0]
			if doc in docs_tlinks.keys():
				docs_tlinks[doc].append((x[1],x[2],x[3]))
			else:
				docs_tlinks[doc] = [(x[1],x[2],x[3])]
	return docs_tlinks

def get_ei_e_mapping(documentpath, doc, docs_tlinks_old):
	xmlfile = open(documentpath + doc + ".tml")
	xmldoc = xmlfile.read()
	soup = BeautifulSoup(xmldoc, 'lxml')
	mapping =  [(x.attrs['eventid'], x.attrs['eiid']) for x in soup.findAll('makeinstance')]
	for word in soup.find("timeml"):
		if str(type(word)) == "<class 'bs4.element.Tag'>":
			tagname = word.name
			tag_text = word.text
			tag_attr = word.attrs
			if tagname == "timex3":
				if tag_attr['functionindocument'] == 'CREATION_TIME':
					dct = tag_attr['tid'] 
	docs_tlinks = []
	for d in docs_tlinks_old:
		x = d[0]
		y = d[1]
		z = d[2]

		if x == dct or y == dct:
			continue
		reset = False
		for m in mapping:
			if x == m[0]:
				x = m[1]
				reset = True
			if y == m[0]:
				y = m[1]
				reset = True
			if "t" in x and "t" in y:
				reset = True
		if not reset:
			continue
		
		if "t" in x or "t" in y:
			docs_tlinks.append((x, y, z))
	return docs_tlinks

def pad_to_length(np_arr, length):
    result = np.zeros(length)
    result[0:np_arr.shape[0]] = np_arr
    return result

def pad_redundant_rows(vec, row_size):
	result = np.zeros((row_size, len(vec)))
	result[0] = vec
	return result	

class Event_Model(object):
	seq_words_x = None
	seq_len_x = None
	e1_idx = None
	hidden_e1 = None
	sess = None
	saver = None
	graph = None
	word_embeddings = None


	def __init__(self, folder_name):
		self.sess =  tf.Session()
		self.saver = tf.train.import_meta_graph(folder_name + '/model.ckpt.meta')
		self.saver.restore(self.sess, tf.train.latest_checkpoint(folder_name))
		self.graph = tf.get_default_graph()

		self.seq_words_x = self.graph.get_tensor_by_name('p1:0')
		self.seq_len_x = self.graph.get_tensor_by_name('p3:0')
		self.e1_idx = self.graph.get_tensor_by_name('p5:0')
		self.hidden_e1 = self.graph.get_tensor_by_name('hidden_e1:0')
		self.seq_words_y = self.graph.get_tensor_by_name('p2:0')
		self.seq_len_y = self.graph.get_tensor_by_name('p4:0')
		self.e2_idx = self.graph.get_tensor_by_name('p6:0')
		self.word_embeddings = read_word_embeddings('../biLSTM_dep/resources/glove.6B.50d-relativized_giga.txt', 50)
		self.W1 = self.graph.get_tensor_by_name('W1:0')
		self.outputlayer = self.graph.get_tensor_by_name('output_layer:0')
		self.layer1 = self.graph.get_tensor_by_name('layer1:0')

	def get_event_vectors(self, e1_idx, tokens):
		words = pad_redundant_rows(pad_to_length(np.array(self.word_embeddings.word_indexer.get_index_list(list(tokens))), 200), 16)
		length1 = np.array([len(list(tokens))] + [0] * 15)
		e1 = np.array([e1_idx] + [0]* 15)

		[output] = self.sess.run([ self.hidden_e1], feed_dict = {self.seq_words_x: words, self.seq_len_x: length1, self.e1_idx : e1})
		return output[0]

	def get_joint_vectors(self, e1_idx, e2_idx, tokens1, tokens2):
		words_x = pad_redundant_rows(pad_to_length(np.array(self.word_embeddings.word_indexer.get_index_list(tokens1)), 200), 16)
		words_y = pad_redundant_rows(pad_to_length(np.array(self.word_embeddings.word_indexer.get_index_list(tokens2)), 200), 16)
		e1 = np.array([e1_idx] + [0]* 15)
		e2 = np.array([e2_idx] + [0]* 15)
		length1 = np.array([len(tokens1)] + [0] * 15)
		length2 = np.array([len(tokens2)] + [0] * 15)
		[outputlayer] = self.sess.run([ self.layer1], feed_dict = {self.seq_words_x: words_x, self.seq_len_x: length1, self.e1_idx : e1,
			self.seq_words_y: words_y, self.seq_len_y: length2, self.e2_idx : e2})

		return outputlayer[0]


	def close_session(self):
		self.sess.close()
		tf.Session().close()




