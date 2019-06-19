import tensorflow as tf
import numpy as np
import random
from utils import *
from utils_temporal import *
from datetime import datetime
from utils_timex import *


def pad_to_length(np_arr, length):
    result = np.zeros(length)
    result[0:np_arr.shape[0]] = np_arr
    return result

def print_f1_scores(pred_output_mat, tag_indexer):

	prec = 0
	rec = 0
	for i in range(len(tag_indexer)):
		try:
			p = float(pred_output_mat[i][i])/float(np.sum(pred_output_mat[i,:]))
		except:
			p = 0
		try:
			r = float(pred_output_mat[i][i])/float(np.sum(pred_output_mat[:,i]))
		except:
			r = 0
		try:
			f1 = (2*p*r)/(p + r)
		except:
			f1 = 0


		print(tag_indexer.get_object(i))
		print("precision: " + str(p))
		print("recall: " + str(r))
		print("f1: " + str(f1))
		prec += p * np.sum(pred_output_mat[:,i])
		rec += r * np.sum(pred_output_mat[:,i])

	prec = float(prec)/float(np.sum(pred_output_mat))
	rec = float(rec)/float(np.sum(pred_output_mat))
	print("overall precision:" + str(prec))
	print("overall recall:" + str(rec))
	print("overall f1:" + str((2*prec*rec)/(prec + rec)))

def hasNumbers(inputString):
	return any(char.isdigit() for char in inputString)

def check_suitability(tokens):
	months = ["January","Jan", "February", "Feb", "March", "Mar", "April", "Apr", "May", "June", "Jun", 
				"July", "Jul", "August", "Aug", "September", "Sep", "October", "Oct", "November", "Nov", "December", "Dec"]
	months = [month.lower() for month in months]

	ret = [False] * len(tokens)
	for i, token in enumerate(tokens):
		if token in months:
			ret[i] = True
		elif hasNumbers(token):
			ret[i] = True
	return ret


def get_timex_vectors(tags, sent, parents, seq_max_len, timex_model):

	vector = np.zeros((seq_max_len,50))
	start_idx = None
	current_t = None
	found = 0
	timex = " "

	for i, (t,w) in enumerate(zip(tags, sent)):
		if "t" not in t or i == len(sent) -1:
			if start_idx != None:
				bool_vec = [True] * len(sent[start_idx:i])
				token = [w for w,b in zip(sent[start_idx:i], bool_vec) if b == True]
				if len(token) == 0:
					continue
				else:
					found += 1
					timex += " ".join(token)
					embed = timex_model.get_timex_embed(" ".join(token), seq_max_len)
					for j in range(start_idx, i):
						vector[j] = embed
				try:
					p = parents[current_t]
					if vector[p].all() == 0:
						vector[p] = embed
					else:
						vector[p] = np.mean([vector[p], embed])
				except:
					x = 1
				
				start_idx = None
		elif "t" in t and current_t != t and current_t != None:
			if start_idx != None:
				bool_vec = [True] * len(sent[start_idx:i])
				token = [w for w,b in zip(sent[start_idx:i], bool_vec) if b == True]
				if len(token) == 0:
					continue
				else:
					timex += " ".join(token)
					embed = timex_model.get_timex_embed(" ".join(token), seq_max_len)
					for j in range(start_idx, i):
						vector[j] = embed
				
				try:
					p = parents[current_t]
					if vector[p].all() == 0:
						vector[p] = embed
					else:
						vector[p] = np.mean([vector[p], embed])
				except:
					x = 1
				
				start_idx = i
				current_t = t
				found += 1
		elif start_idx == None and i != len(sent) - 1:
				start_idx = i
				current_t = t


	return vector


def build_model(train_exs, test_exs, word_vectors, tag_indexer, output_folder):

	seq_max_len = 200
	seq_dep_max_len = 20

	timex_model = Timex_Model("../timex_model/model_timex_pairs")

	train_sent_idx_x = np.asarray([pad_to_length(np.array(word_vectors.word_indexer.get_index_list(ex.sent1)), seq_max_len) for ex in train_exs])
	train_sent_idx_y = np.asarray([pad_to_length(np.array(word_vectors.word_indexer.get_index_list(ex.sent2)), seq_max_len) for ex in train_exs])
	train_timex_x = np.asarray([get_timex_vectors(ex.tags1, ex.sent1, ex.tid_parents1, seq_max_len, timex_model) for ex in train_exs])
	train_timex_y = np.asarray([get_timex_vectors(ex.tags2, ex.sent2, ex.tid_parents2, seq_max_len, timex_model) for ex in train_exs])
	train_dep_path_x = np.asarray([pad_to_length(np.array(ex.dep_path1), seq_dep_max_len) for ex in train_exs])
	train_dep_path_y = np.asarray([pad_to_length(np.array(ex.dep_path2), seq_dep_max_len) for ex in train_exs])
	train_seq_lens_x = np.array([len(ex.sent1) for ex in train_exs])
	train_seq_lens_y = np.array([len(ex.sent2) for ex in train_exs])
	train_dep_path_lens_x = np.array([len(ex.dep_path1) for ex in train_exs])
	train_dep_path_lens_y = np.array([len(ex.dep_path2) for ex in train_exs])
	train_labels_arr = np.array([tag_indexer.get_index(ex.label) for ex in train_exs])



	test_sent_idx_x = np.asarray([pad_to_length(np.array(word_vectors.word_indexer.get_index_list(ex.sent1)), seq_max_len) for ex in test_exs])
	test_sent_idx_y = np.asarray([pad_to_length(np.array(word_vectors.word_indexer.get_index_list(ex.sent2)), seq_max_len) for ex in test_exs])
	test_timex_x = np.asarray([get_timex_vectors(ex.tags1, ex.sent1, ex.tid_parents1, seq_max_len, timex_model) for ex in test_exs])
	test_timex_y = np.asarray([get_timex_vectors(ex.tags2, ex.sent2, ex.tid_parents2, seq_max_len, timex_model) for ex in test_exs])
	test_dep_path_x = np.asarray([pad_to_length(np.array(ex.dep_path1), seq_dep_max_len) for ex in test_exs])
	test_dep_path_y = np.asarray([pad_to_length(np.array(ex.dep_path2), seq_dep_max_len) for ex in test_exs])
	test_seq_lens_x = np.array([len(ex.sent1) for ex in test_exs])
	test_seq_lens_y = np.array([len(ex.sent2) for ex in test_exs])
	test_dep_path_lens_x = np.array([len(ex.dep_path1) for ex in test_exs])
	test_dep_path_lens_y = np.array([len(ex.dep_path2) for ex in test_exs])
	test_labels_arr = np.array([tag_indexer.get_index(ex.label) for ex in test_exs])
	
	timex_model.close_session()
	tf.reset_default_graph()

	word_embeddings = np.array(word_vectors.vectors, dtype="float32")

	num_classes = len(tag_indexer)
	lstm_cell_size = 300
	batch_size = 16

	seq_x = tf.placeholder(tf.int32, [batch_size ,seq_max_len], name= "seq_x")
	seq_y = tf.placeholder(tf.int32, [batch_size ,seq_max_len], name = "seq_y")
	dep_path_x = tf.placeholder(tf.int32, [batch_size, seq_dep_max_len], name = "dep_path_x")
	dep_path_y = tf.placeholder(tf.int32, [batch_size, seq_dep_max_len], name = "dep_path_y")
	seq_len_x = tf.placeholder(tf.int32, batch_size, name = "seq_len_x")
	seq_len_y = tf.placeholder(tf.int32, batch_size, name = "seq_len_y")
	seq_dep_len_x = tf.placeholder(tf.int32, batch_size, name = "seq_dep_len_x")
	seq_dep_len_y = tf.placeholder(tf.int32, batch_size, name = "seq_dep_len_y")
	label = tf.placeholder(tf.int32, batch_size, name = "label")

	timex_x = tf.placeholder(tf.float32, [batch_size, seq_max_len, 50], name = "timex_x")
	timex_y = tf.placeholder(tf.float32, [batch_size, seq_max_len, 50], name = "timex_y")

	words_x = tf.nn.embedding_lookup(word_embeddings,seq_x)
	words_y = tf.nn.embedding_lookup(word_embeddings,seq_y)

	input_x = tf.concat([words_x, timex_x], axis = 2)
	input_y = tf.concat([words_y, timex_y], axis = 2)

	with tf.variable_scope('lower'):
		cellf_lower = tf.nn.rnn_cell.LSTMCell(num_units=lstm_cell_size, state_is_tuple=True)
		cellb_lower = tf.nn.rnn_cell.LSTMCell(num_units=lstm_cell_size, state_is_tuple=True)

		bi_outputs_x, _ = tf.nn.bidirectional_dynamic_rnn(cellf_lower, cellb_lower, input_x, dtype = tf.float32, sequence_length = seq_len_x)
		bi_outputs_y, _ = tf.nn.bidirectional_dynamic_rnn(cellf_lower, cellb_lower, input_y, dtype = tf.float32, sequence_length = seq_len_y)
	
	bi_outputs_x_concat = tf.concat([bi_outputs_x[0],bi_outputs_x[1]], axis = 2)
	bi_outputs_y_concat = tf.concat([bi_outputs_y[0],bi_outputs_y[1]], axis = 2)


	idx = [[i]*seq_dep_max_len for i in range(batch_size)]
	idx1 = tf.stack([idx, dep_path_x], axis = 2)
	idx2 = tf.stack([idx, dep_path_y], axis = 2)

	second_input_x = tf.gather_nd(bi_outputs_x_concat, idx1)
	second_input_y = tf.gather_nd(bi_outputs_y_concat, idx2)


	with tf.variable_scope('upper'):
		cellf_upper = tf.nn.rnn_cell.LSTMCell(num_units=lstm_cell_size, state_is_tuple=True)
		cellb_upper = tf.nn.rnn_cell.LSTMCell(num_units=lstm_cell_size, state_is_tuple=True)
		second_bi_outputs_x, state_x = tf.nn.bidirectional_dynamic_rnn(cellf_upper, cellb_upper, second_input_x, dtype = tf.float32, sequence_length = seq_dep_len_x)
		second_bi_outputs_y, state_y = tf.nn.bidirectional_dynamic_rnn(cellf_upper, cellb_upper, second_input_y, dtype = tf.float32, sequence_length = seq_dep_len_y)

	indices_x = tf.stack([tf.range(batch_size), seq_dep_len_x - 1], axis=1)
	indices_y = tf.stack([tf.range(batch_size), seq_dep_len_y - 1], axis=1)

	a = state_x[0][1]
	b = state_x[1][1]
	c = state_y[0][1]
	d = state_y[1][1]

	output = tf.concat([a, b, c, d], axis = 1)


	W1 = tf.get_variable("W1", [ output.shape[1], lstm_cell_size], initializer=tf.contrib.layers.xavier_initializer(seed=0))
	layer1 = tf.nn.tanh(tf.tensordot(output, W1, 1))

	W = tf.get_variable("W", [ lstm_cell_size , num_classes], initializer=tf.contrib.layers.xavier_initializer(seed=0))
	output_layer = tf.nn.tanh(tf.tensordot(layer1, W, 1))

	output_layer = tf.identity(output_layer, name="output_layer")
	loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = output_layer, labels = label)
	loss = tf.reduce_mean(loss)


	domain_pred = tf.argmax(output_layer,axis=1)
	domain_pred = tf.identity(domain_pred, name="domain_pred")

	decay_steps = 100
	learning_rate_decay_factor = .9995
	global_step = tf.contrib.framework.get_or_create_global_step()
	initial_learning_rate = 0.0001


	lr = tf.train.exponential_decay(initial_learning_rate, global_step, decay_steps, learning_rate_decay_factor, staircase=True)    
	optimizer = tf.train.AdamOptimizer(lr)
	grads = optimizer.compute_gradients(loss)
	apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)
	with tf.control_dependencies([apply_gradient_op]):
		train_op = tf.no_op(name='train')

	init = tf.global_variables_initializer()
	num_epochs = 30
	
	saver = tf.train.Saver()

	with tf.Session() as sess1:
		tf.set_random_seed(0)
		sess1.run(init)


		for i in range(0, num_epochs):
			pred_output_mat = np.zeros((len(tag_indexer), len(tag_indexer)))
			print("Epoch#"+str(i) + " started")
			acc = 0
			total = 0
			acc_train = 0
			total_train = 0
			loss_this_iter = 0
			time_now = datetime.now()

			for j in range(int(len(train_sent_idx_x)/batch_size)):
				indices = random.sample(range(len(train_sent_idx_x)), batch_size)
				
				[ _, loss_this_instance, domain_predicted, output_test] = sess1.run([ train_op, loss, domain_pred, output_layer], feed_dict = {seq_x: train_sent_idx_x[indices], seq_y: train_sent_idx_y[indices],
													seq_len_x: train_seq_lens_x[indices], seq_len_y: train_seq_lens_y[indices], label: train_labels_arr[indices],
													dep_path_x: train_dep_path_x[indices], dep_path_y:train_dep_path_y[indices],
													seq_dep_len_x: train_dep_path_lens_x[indices], seq_dep_len_y: train_dep_path_lens_y[indices], 
													timex_x: train_timex_x[indices], timex_y: train_timex_y[indices]})

				loss_this_iter += loss_this_instance

			print("loss:" + repr(loss_this_iter) + "\n")

			
			for ex_idx in range(0, int(len(dev_sent_idx_x)/batch_size)):
				indices = range(batch_size*ex_idx, batch_size*ex_idx + batch_size)
				[predicted, output_test] = sess1.run([ domain_pred, output_layer], feed_dict = {seq_x: dev_sent_idx_x[indices], seq_y: dev_sent_idx_y[indices],
													seq_len_x: dev_seq_lens_x[indices], seq_len_y: dev_seq_lens_y[indices],
													dep_path_x: dev_dep_path_x[indices], dep_path_y:dev_dep_path_y[indices],
													seq_dep_len_x: dev_dep_path_lens_x[indices], seq_dep_len_y: dev_dep_path_lens_y[indices],
													timex_x: dev_timex_x[indices], timex_y: dev_timex_y[indices]})

				actual = dev_labels_arr[indices]
				

				for idx in range(len(actual)):
					pred_output_mat[predicted[idx]][actual[idx]] += 1
					if predicted[idx] == actual[idx]:
						acc += 1
					total += 1
			
			for ex_idx in range(0, int(len(train_sent_idx_x)/batch_size)):
				indices = range(batch_size*ex_idx, batch_size*ex_idx + batch_size)
				[predicted] = sess1.run([ domain_pred], feed_dict = {seq_x: train_sent_idx_x[indices], seq_y: train_sent_idx_y[indices],
													seq_len_x: train_seq_lens_x[indices], seq_len_y: train_seq_lens_y[indices],
													dep_path_x: train_dep_path_x[indices], dep_path_y:train_dep_path_y[indices],
													seq_dep_len_x: train_dep_path_lens_x[indices], seq_dep_len_y: train_dep_path_lens_y[indices],
													timex_x: train_timex_x[indices], timex_y: train_timex_y[indices]})

				actual = train_labels_arr[indices]

				for idx in range(len(actual)):
					if predicted[idx] == actual[idx]:
						acc_train += 1
					total_train += 1

			print("train accuracy for epoch: " + str(float(acc_train)/float(total_train)))
			print("dev accuracy for epoch: " + str(float(acc)/float(total)))

		save_path = saver.save(sess1, os.path.join(output_folder, "model.ckpt"))
		print("Model saved in path: %s" % save_path)


def pad_to_length_2d(input_, max_len):
	output = np.zeros((max_len, input_.shape[1]))
	for i in range(input_.shape[0]):
		output[i] = input_[i]

	return output

def build_model_elmo(train_exs, dev_exs, test_exs, word_vectors, tag_indexer, pos_indexer, dep_indexer, output_folder):

	seq_max_len = 200
	seq_dep_max_len = 20

	print("preparing data..")
	timex_model = Timex_Model("../timex_model/model_timex_pairs")


	train_timex_x = np.asarray([get_timex_vectors(ex.tags1, ex.sent1, ex.tid_parents1, seq_max_len, timex_model) for ex in train_exs])
	train_timex_y = np.asarray([get_timex_vectors(ex.tags2, ex.sent2, ex.tid_parents2, seq_max_len, timex_model) for ex in train_exs])

	test_timex_x = np.asarray([get_timex_vectors(ex.tags1, ex.sent1, ex.tid_parents1, seq_max_len, timex_model) for ex in test_exs])
	test_timex_y = np.asarray([get_timex_vectors(ex.tags2, ex.sent2, ex.tid_parents2, seq_max_len, timex_model) for ex in test_exs])

	timex_model.close_session()
	tf.reset_default_graph()

	num_classes = len(tag_indexer)
	lstm_cell_size = 50
	batch_size = 16

	seq_x = tf.placeholder(tf.float32, [batch_size ,seq_max_len, train_exs[0].elmo1.shape[1]], name= "seq_x")
	seq_y = tf.placeholder(tf.float32, [batch_size ,seq_max_len, train_exs[0].elmo1.shape[1]], name = "seq_y")
	dep_path_x = tf.placeholder(tf.int32, [batch_size, seq_dep_max_len], name = "dep_path_x")
	dep_path_y = tf.placeholder(tf.int32, [batch_size, seq_dep_max_len], name = "dep_path_y")
	seq_len_x = tf.placeholder(tf.int32, batch_size, name = "seq_len_x")
	seq_len_y = tf.placeholder(tf.int32, batch_size, name = "seq_len_y")
	seq_dep_len_x = tf.placeholder(tf.int32, batch_size, name = "seq_dep_len_x")
	seq_dep_len_y = tf.placeholder(tf.int32, batch_size, name = "seq_dep_len_y")
	label = tf.placeholder(tf.int32, batch_size, name = "label")

	timex_x = tf.placeholder(tf.float32, [batch_size, seq_max_len, 50], name = "timex_x")
	timex_y = tf.placeholder(tf.float32, [batch_size, seq_max_len, 50], name = "timex_y")

	input_x = tf.concat([seq_x, timex_x], axis = 2)
	input_y = tf.concat([seq_y, timex_y], axis = 2)

	with tf.variable_scope('lower'):
		cellf_lower = tf.nn.rnn_cell.LSTMCell(num_units=lstm_cell_size, state_is_tuple=True)
		cellb_lower = tf.nn.rnn_cell.LSTMCell(num_units=lstm_cell_size, state_is_tuple=True)

		bi_outputs_x, _ = tf.nn.bidirectional_dynamic_rnn(cellf_lower, cellb_lower, input_x, dtype = tf.float32, sequence_length = seq_len_x)
		bi_outputs_y, _ = tf.nn.bidirectional_dynamic_rnn(cellf_lower, cellb_lower, input_y, dtype = tf.float32, sequence_length = seq_len_y)
	
	bi_outputs_x_concat = tf.concat([bi_outputs_x[0],bi_outputs_x[1]], axis = 2)
	bi_outputs_y_concat = tf.concat([bi_outputs_y[0],bi_outputs_y[1]], axis = 2)

	idx = [[i]*seq_dep_max_len for i in range(batch_size)]
	idx1 = tf.stack([idx, dep_path_x], axis = 2)
	idx2 = tf.stack([idx, dep_path_y], axis = 2)

	second_input_x = tf.gather_nd(bi_outputs_x_concat, idx1)
	second_input_y = tf.gather_nd(bi_outputs_y_concat, idx2)


	with tf.variable_scope('upper'):
		cellf_upper = tf.nn.rnn_cell.LSTMCell(num_units=lstm_cell_size, state_is_tuple=True)
		cellb_upper = tf.nn.rnn_cell.LSTMCell(num_units=lstm_cell_size, state_is_tuple=True)
		second_bi_outputs_x, state_x = tf.nn.bidirectional_dynamic_rnn(cellf_upper, cellb_upper, second_input_x, dtype = tf.float32, sequence_length = seq_dep_len_x)
		second_bi_outputs_y, state_y = tf.nn.bidirectional_dynamic_rnn(cellf_upper, cellb_upper, second_input_y, dtype = tf.float32, sequence_length = seq_dep_len_y)

	indices_x = tf.stack([tf.range(batch_size), seq_dep_len_x - 1], axis=1)
	indices_y = tf.stack([tf.range(batch_size), seq_dep_len_y - 1], axis=1)

	a = state_x[0][1]
	b = state_x[1][1]
	c = state_y[0][1]
	d = state_y[1][1]

	output = tf.concat([a, b, c, d], axis = 1)


	W1 = tf.get_variable("W1", [ output.shape[1], lstm_cell_size], initializer=tf.contrib.layers.xavier_initializer(seed=0))
	layer1 = tf.nn.tanh(tf.tensordot(output, W1, 1))

	W = tf.get_variable("W", [ lstm_cell_size , num_classes], initializer=tf.contrib.layers.xavier_initializer(seed=0))
	output_layer = tf.nn.tanh(tf.tensordot(layer1, W, 1))

	output_layer = tf.identity(output_layer, name="output_layer")
	loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = output_layer, labels = label)
	loss = tf.reduce_mean(loss)


	domain_pred = tf.argmax(output_layer,axis=1)
	domain_pred = tf.identity(domain_pred, name="domain_pred")

	decay_steps = 10
	learning_rate_decay_factor = .999995
	global_step = tf.contrib.framework.get_or_create_global_step()
	initial_learning_rate = 0.001


	lr = tf.train.exponential_decay(initial_learning_rate, global_step, decay_steps, learning_rate_decay_factor, staircase=True)    
	optimizer = tf.train.AdamOptimizer(lr)
	grads = optimizer.compute_gradients(loss)
	apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)
	with tf.control_dependencies([apply_gradient_op]):
		train_op = tf.no_op(name='train')

	init = tf.global_variables_initializer()
	num_epochs = 10
	
	saver = tf.train.Saver()

	if not os.path.exists(output_folder):
		os.mkdir(output_folder)

	with tf.Session() as sess1:
		tf.set_random_seed(0)
		sess1.run(init)
		for i in range(0, num_epochs):
			pred_output_mat = np.zeros((len(tag_indexer), len(tag_indexer)))
			print("Epoch#"+str(i) + " started")
			acc = 0
			total = 0
			acc_train = 0
			total_train = 0
			loss_this_iter = 0
			time_now = datetime.now()

			for j in range(int(len(train_exs)/batch_size)):
				indices = random.sample(range(len(train_exs)), batch_size)


				input_sent_idx_x = np.asarray([pad_to_length_2d(train_exs[idx].elmo1, seq_max_len) for idx in indices])
				input_sent_idx_y = np.asarray([pad_to_length_2d(train_exs[idx].elmo2, seq_max_len) for idx in indices])
				input_seq_len_x = np.array([len(train_exs[idx].sent1) for idx in indices])
				input_seq_len_y = np.array([len(train_exs[idx].sent2) for idx in indices])
				input_labels = np.array([tag_indexer.get_index(train_exs[idx].label) for idx in indices])
				input_dep_path_x = np.asarray([pad_to_length(np.array(train_exs[idx].dep_path1), seq_dep_max_len) for idx in indices])
				input_dep_path_y = np.asarray([pad_to_length(np.array(train_exs[idx].dep_path2), seq_dep_max_len) for idx in indices])
				input_seq_dep_len_x = np.array([len(train_exs[idx].dep_path1) for idx in indices])
				input_seq_dep_len_y = np.array([len(train_exs[idx].dep_path2) for idx in indices])

				[ _, loss_this_instance, domain_predicted, output_test] = sess1.run([ train_op, loss, domain_pred, output_layer], feed_dict = 
													{seq_x: input_sent_idx_x, seq_y: input_sent_idx_y,
													seq_len_x: input_seq_len_x, seq_len_y: input_seq_len_y, 
													label: input_labels,
													dep_path_x: input_dep_path_x, dep_path_y: input_dep_path_y,
													seq_dep_len_x: input_seq_dep_len_x, seq_dep_len_y: input_seq_dep_len_y
													,timex_x: train_timex_x[indices], timex_y: train_timex_y[indices]})

				loss_this_iter += loss_this_instance

			print("loss:" + repr(loss_this_iter) + "\n")

			for ex_idx in range(0, int(len(train_exs)/batch_size)):
				indices = range(batch_size*ex_idx, batch_size*ex_idx + batch_size)
				input_sent_idx_x = np.asarray([pad_to_length_2d(train_exs[idx].elmo1, seq_max_len) for idx in indices])
				input_sent_idx_y = np.asarray([pad_to_length_2d(train_exs[idx].elmo2, seq_max_len) for idx in indices])
				input_seq_len_x = np.array([len(train_exs[idx].sent1) for idx in indices])
				input_seq_len_y = np.array([len(train_exs[idx].sent2) for idx in indices])
				input_labels = np.array([tag_indexer.get_index(train_exs[idx].label) for idx in indices])
				input_dep_path_x = np.asarray([pad_to_length(np.array(train_exs[idx].dep_path1), seq_dep_max_len) for idx in indices])
				input_dep_path_y = np.asarray([pad_to_length(np.array(train_exs[idx].dep_path2), seq_dep_max_len) for idx in indices])
				input_seq_dep_len_x = np.array([len(train_exs[idx].dep_path1) for idx in indices])
				input_seq_dep_len_y = np.array([len(train_exs[idx].dep_path2) for idx in indices])

				[predicted] = sess1.run([ domain_pred], feed_dict = {seq_x: input_sent_idx_x, seq_y: input_sent_idx_y,
													seq_len_x: input_seq_len_x, seq_len_y: input_seq_len_y, 
													dep_path_x: input_dep_path_x, dep_path_y: input_dep_path_y,
													seq_dep_len_x: input_seq_dep_len_x, seq_dep_len_y: input_seq_dep_len_y
													,timex_x: train_timex_x[indices], timex_y: train_timex_y[indices]})

				actual = input_labels

				for idx in range(len(actual)):
					if predicted[idx] == actual[idx]:
						acc_train += 1
					total_train += 1

			print("train accuracy for epoch: " + str(float(acc_train)/float(total_train)))

			pred_output_mat = np.zeros((len(tag_indexer), len(tag_indexer)))
			for ex_idx in range(0, int(len(test_exs)/batch_size)):
				indices = range(batch_size*ex_idx, batch_size*ex_idx + batch_size)

				input_sent_idx_x = np.asarray([pad_to_length_2d(test_exs[idx].elmo1, seq_max_len) for idx in indices])
				input_sent_idx_y = np.asarray([pad_to_length_2d(test_exs[idx].elmo2, seq_max_len) for idx in indices])
				input_seq_len_x = np.array([len(test_exs[idx].sent1) for idx in indices])
				input_seq_len_y = np.array([len(test_exs[idx].sent2) for idx in indices])
				input_labels = np.array([tag_indexer.get_index(test_exs[idx].label) for idx in indices])
				input_dep_path_x = np.asarray([pad_to_length(np.array(test_exs[idx].dep_path1), seq_dep_max_len) for idx in indices])
				input_dep_path_y = np.asarray([pad_to_length(np.array(test_exs[idx].dep_path2), seq_dep_max_len) for idx in indices])
				input_seq_dep_len_x = np.array([len(test_exs[idx].dep_path1) for idx in indices])
				input_seq_dep_len_y = np.array([len(test_exs[idx].dep_path2) for idx in indices])

				[predicted] = sess1.run([ domain_pred], feed_dict = {seq_x: input_sent_idx_x, seq_y: input_sent_idx_y,
													seq_len_x: input_seq_len_x, seq_len_y: input_seq_len_y, 
													dep_path_x: input_dep_path_x, dep_path_y: input_dep_path_y,
													seq_dep_len_x: input_seq_dep_len_x, seq_dep_len_y: input_seq_dep_len_y
													,timex_x: test_timex_x[indices], timex_y: test_timex_y[indices]})

				actual = input_labels

				for idx in range(len(actual)):
					pred_output_mat[predicted[idx]][actual[idx]] += 1
					if predicted[idx] == actual[idx]:
						acc += 1
					total += 1


			print("dev accuracy: " + str(float(acc)/float(total)))
			print(pred_output_mat)

		save_path = saver.save(sess1, os.path.join(output_folder, "model.ckpt"))
		print("Model saved in path: %s" % save_path)
