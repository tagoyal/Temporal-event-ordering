from utils_temporal import IndexedExample
from os import listdir
import json, random
from utils import *
from ilp_utils import *
from models import *
import tensorflow as tf
import pickle as pk
import glpk
from itertools import combinations


word_vectors_path = "../code/resources/glove.6B.300d-relativized.txt"
input_folder = "./model_matres_9/"
tlinks_file = '../data/timebank_1_2/TimebankDense.full.txt'
documentpath = "../data/timebank_1_2/data/timeml/"
docs_tlinks_old = get_tlinks(tlinks_file)
docs_tlinks = {}
for doc in docs_tlinks_old.keys():
	docs_tlinks[doc] = get_ei_e_mapping(documentpath, doc, docs_tlinks_old[doc])

def get_softmax(x):
	#e_x = np.exp(x - np.max(x))
	#return e_x / e_x.sum()
	return np.exp(x)/np.sum(np.exp(x))


class PredictedInstance:
	def __init__(self, sent1, sent2, w1, w2, predicted, actual, softmax, doc, e1, e2, dep_path1, dep_path2):
		self.sent1 = sent1
		self.sent2 = sent2
		self.w1 = w1
		self.w2 = w2
		self.predicted = predicted
		self.actual = actual
		self.softmax = softmax
		self.doc = doc
		self.e1 = e1
		self.e2 = e2
		self.dep_path1 = dep_path1
		self.dep_path2 = dep_path2

if __name__ == '__main__':
	word_vectors = read_word_embeddings(word_vectors_path)
	seq_max_len = 200
	seq_dep_max_len = 20
	batch_size = 16

	tag_indexer = pk.load(open(os.path.join(input_folder, "tag_indexer.pkl"), "rb"))
	dep_indexer = pk.load(open(os.path.join(input_folder, "deps_indexer.pkl")))
	pos_indexer = pk.load(open(os.path.join(input_folder, "pos_indexer.pkl"), "rb"))
	dev_exs = pk.load(open(os.path.join(input_folder, "test_data.pkl")))
	train_exs = pk.load(open(os.path.join(input_folder, "train_data.pkl")))

	dev_predicted = []
	with tf.Session() as sess:
		saver = tf.train.import_meta_graph(os.path.join(input_folder, 'model.ckpt.meta'))
		saver.restore(sess, tf.train.latest_checkpoint(input_folder))
		graph = tf.get_default_graph()

		seq_x = graph.get_tensor_by_name("p1:0")
		seq_y = graph.get_tensor_by_name("p2:0")
		seq_pos_x = graph.get_tensor_by_name("p3:0")
		seq_pos_y = graph.get_tensor_by_name("p4:0")
		seq_rel_x = graph.get_tensor_by_name("p5:0")
		seq_rel_y = graph.get_tensor_by_name("p6:0")
		dep_path_x = graph.get_tensor_by_name("p7:0")
		dep_path_y = graph.get_tensor_by_name("p8:0")
		seq_len_x = graph.get_tensor_by_name("p9:0")
		seq_len_y = graph.get_tensor_by_name("p10:0")
		seq_dep_len_x = graph.get_tensor_by_name("p11:0")
		seq_dep_len_y = graph.get_tensor_by_name("p12:0")
		domain_pred =  graph.get_tensor_by_name('domain_pred:0')
		output_layer =  graph.get_tensor_by_name('output_layer:0')

		dev_sent_idx_x = np.asarray([pad_to_length(np.array(word_vectors.word_indexer.get_index_list(ex.sent1)), seq_max_len) for ex in dev_exs])
		dev_sent_idx_y = np.asarray([pad_to_length(np.array(word_vectors.word_indexer.get_index_list(ex.sent2)), seq_max_len) for ex in dev_exs])
		dev_pos_x = np.asarray([pad_to_length(np.array(pos_indexer.get_index_list(ex.pos1)), seq_max_len) for ex in dev_exs])
		dev_pos_y = np.asarray([pad_to_length(np.array(pos_indexer.get_index_list(ex.pos2)), seq_max_len) for ex in dev_exs])
		dev_deps_x = np.asarray([pad_to_length(np.array(dep_indexer.get_index_list(ex.rel1)), seq_max_len) for ex in dev_exs])
		dev_deps_y = np.asarray([pad_to_length(np.array(dep_indexer.get_index_list(ex.rel2)), seq_max_len) for ex in dev_exs])
		dev_dep_path_x = np.asarray([pad_to_length(np.array(ex.dep_path1), seq_dep_max_len) for ex in dev_exs])
		dev_dep_path_y = np.asarray([pad_to_length(np.array(ex.dep_path2), seq_dep_max_len) for ex in dev_exs])
		dev_seq_lens_x = np.array([len(ex.sent1) for ex in dev_exs])
		dev_seq_lens_y = np.array([len(ex.sent2) for ex in dev_exs])
		dev_dep_path_lens_x = np.array([len(ex.dep_path1) for ex in dev_exs])
		dev_dep_path_lens_y = np.array([len(ex.dep_path2) for ex in dev_exs])
		dev_labels_arr = np.array([tag_indexer.get_index(ex.label) for ex in dev_exs])
		doc_id = [ex.doc_name for ex in dev_exs]
		e1_ids = [ex.eid1 for ex in dev_exs]
		e2_ids = [ex.eid2 for ex in dev_exs]
		dev_sent1 = [" ".join(ex.sent1) for ex in dev_exs]
		dev_sent2 = [" ".join(ex.sent2) for ex in dev_exs]

		for ex_idx in range(0, int(len(dev_sent_idx_x)/batch_size)):
			indices = range(batch_size*ex_idx, batch_size*ex_idx + batch_size)
			[softmax, predicted] = sess.run([output_layer, domain_pred], feed_dict = {seq_x: dev_sent_idx_x[indices], seq_y: dev_sent_idx_y[indices],
												seq_len_x: dev_seq_lens_x[indices], seq_len_y: dev_seq_lens_y[indices],
												seq_pos_x: dev_pos_x[indices], seq_pos_y: dev_pos_y[indices], seq_rel_x: dev_deps_x[indices],
												seq_rel_y: dev_deps_y[indices], dep_path_x: dev_dep_path_x[indices], dep_path_y:dev_dep_path_y[indices],
												seq_dep_len_x: dev_dep_path_lens_x[indices], seq_dep_len_y: dev_dep_path_lens_y[indices]})

			actual = dev_labels_arr[indices]

			for i in range(batch_size):
				w1 = word_vectors.word_indexer.get_object(dev_sent_idx_x[indices[i]][int(dev_dep_path_x[indices[i]][0])])
				w2 = word_vectors.word_indexer.get_object(dev_sent_idx_y[indices[i]][int(dev_dep_path_y[indices[i]][0])])
				dev_predicted.append(PredictedInstance(dev_sent1[indices[i]], dev_sent2[indices[i]], w1, w2, tag_indexer.get_object(predicted[i]), tag_indexer.get_object(actual[i]), get_softmax(softmax[i]), doc_id[indices[i]],
									e1_ids[indices[i]], e2_ids[indices[i]], dev_dep_path_x[indices[i]], dev_dep_path_y[indices[i]]))



	unique_docs = list(set([d.doc for d in dev_predicted]))
	

	predicted = []
	corrected = 0
	wronged = 0
	total = 0
	neither = 0
	skipped = 0
	kept = 0
	for doc_id in unique_docs:
		additional_tlinks = docs_tlinks[doc_id]

		dev_doc = []
		for d in dev_predicted:
			if d.doc == doc_id:
				dev_doc.append(d)
		for t in additional_tlinks:
			if t[2] == "ii" or t[2] == "i": 
				skipped += 1
				continue
			soft = np.array([0, 0, 0, 0])
			soft[tag_indexer.get_index(t[2])] = 1
			kept += 1
			dev_doc.append(PredictedInstance(None, None, None, None, t[2], t[2], soft, doc_id, t[0], t[1], None, None))
		predicted = get_global_temporal_links(dev_doc, tag_indexer)

		for p in predicted:
			if p[0] != p[2]:
				total += 1
				if p[0] == p[1]:
					corrected += 1
				elif p[1] == p[2]:
					wronged += 1
				else:
					neither += 1
	print "Corrected: " + str(float(corrected)/float(total))
	print "Wronged: " + str(float(wronged)/float(total))
	print "neither: " + str(float(neither)/float(total))
	print total
	print len(dev_predicted)
	print skipped
	print kept
	exit()
	transitivity_checks_on_gold(dev_exs, tag_indexer, word_vectors.word_indexer)
	transitivity_checks_on_gold(train_exs, tag_indexer, word_vectors.word_indexer)

	"""
	def is_substring(a,b):
		t1 = set(a).issubset(set(b))
		t2 = set(b).issubset(set(a))

		return t1 or t2

	mat = [[0,0],[0,0]]
	for doc_id in unique_docs:
		dev_doc = []
		for d in dev_predicted:
			if d.doc == doc_id:
				dev_doc.append(d)

		unique_sents_pairs = []
		for d in dev_doc:
			if (d.sent1, d.sent2) not in unique_sents_pairs:
				unique_sents_pairs.append((d.sent1, d.sent2))

		
		for sent1, sent2 in unique_sents_pairs:
			pred_labels = []
			actual_labels = []
			event_pairs = {}
			for d in dev_doc:
				if d.sent1 == sent1 and d.sent2 == sent2:
					if d.e1 not in event_pairs:
						event_pairs[d.e1] = []
					event_pairs[d.e1].append((d.predicted, d.actual, d.dep_path2))

			for event in event_pairs.keys():
				if len(event_pairs[event]) < 2:
				 	continue
				for a1, a2 in combinations(event_pairs[event], 2):
					if is_substring(a1[2], a2[2]):
						w = len(set(a1[2])^set(a2[2]))
						if w < 2:
							if a1[0] == a2[0]:
								x = 0
							else:
								x = 1
							if a1[1] == a2[1]:
								y = 0
							else:
								y = 1

							mat[x][y] += 1

	print mat

			pred_labels = []
			actual_labels = []
			for d in dev_doc:
				if d.sent1 == sent1 and d.sent2 == sent2:
					pred_labels.append(d.predicted)
					actual_labels.append(d.actual)

			print pred_labels
			print actual_labels
		print "########"
	"""





