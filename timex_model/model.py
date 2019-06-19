import os
import random
import numpy as np
import json
from utils_timex import *
import tensorflow as tf
from datetime import datetime


def pad_to_length(np_arr, length):
    result = np.zeros(length)
    result[0:np_arr.shape[0]] = np_arr
    return result

def print_f1_scores(pred_output_mat, num_classes):

	for i in range(num_classes):
		
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


		print(i)
		print("precision: " + str(p))
		print("recall: " + str(r))
		print("f1: " + str(f1))

def build_model(train_exs, dev_exs, char_embeddings, output_folder):
	seq_len_max = 50


	train_timex1 = np.asarray([pad_to_length(np.array(char_embeddings.word_indexer.get_index_list(ex.timex1_chars)), seq_len_max) for ex in train_exs])
	train_timex2 = np.asarray([pad_to_length(np.array(char_embeddings.word_indexer.get_index_list(ex.timex2_chars)), seq_len_max) for ex in train_exs])
	train_length1 = np.array([len(ex.timex1_chars) for ex in train_exs])
	train_length2 = np.array([len(ex.timex2_chars) for ex in train_exs])
	train_label = np.array([ex.label for ex in train_exs])

	dev_timex1 = np.asarray([pad_to_length(np.array(char_embeddings.word_indexer.get_index_list(ex.timex1_chars)), seq_len_max) for ex in dev_exs])
	dev_timex2 = np.asarray([pad_to_length(np.array(char_embeddings.word_indexer.get_index_list(ex.timex2_chars)), seq_len_max) for ex in dev_exs])
	dev_length1 = np.array([len(ex.timex1_chars) for ex in dev_exs])
	dev_length2 = np.array([len(ex.timex2_chars) for ex in dev_exs])
	dev_label = np.array([ex.label for ex in dev_exs])

	word_embeddings = np.array(char_embeddings.vectors, dtype="float32")
	num_classes = 2
	lstm_cell_size = 25
	batch_size = 10

	seq_x = tf.placeholder(tf.int32, [batch_size ,seq_len_max], name= "p1")
	seq_y = tf.placeholder(tf.int32, [batch_size ,seq_len_max], name = "p2")
	seq_len_x = tf.placeholder(tf.int32, batch_size, name = "p3")
	seq_len_y = tf.placeholder(tf.int32, batch_size, name = "p4")
	label = tf.placeholder(tf.int32, [batch_size], name = "label")

	words_x = tf.nn.embedding_lookup(word_embeddings,seq_x)
	words_y = tf.nn.embedding_lookup(word_embeddings,seq_y)

	with tf.variable_scope("layer1"):
		cellf1 = tf.nn.rnn_cell.LSTMCell(num_units=lstm_cell_size, state_is_tuple=True)
		cellb1 = tf.nn.rnn_cell.LSTMCell(num_units=lstm_cell_size, state_is_tuple=True)

		bi_outputs_x_, state_x = tf.nn.bidirectional_dynamic_rnn(cellf1, cellb1, words_x, dtype = tf.float32, sequence_length = seq_len_x)
		bi_outputs_y_, state_y = tf.nn.bidirectional_dynamic_rnn(cellf1, cellb1, words_y, dtype = tf.float32, sequence_length = seq_len_y)

	
	bi_outputs_x = tf.concat([bi_outputs_x_[0], bi_outputs_x_[1]], axis = 2)
	bi_outputs_y = tf.concat([bi_outputs_y_[0], bi_outputs_y_[1]], axis = 2)

	with tf.variable_scope("layer2"):
		cellf2 = tf.nn.rnn_cell.LSTMCell(num_units=lstm_cell_size, state_is_tuple=True)
		cellb2 = tf.nn.rnn_cell.LSTMCell(num_units=lstm_cell_size, state_is_tuple=True)

		bi_outputs_x_2_, state_x_2 = tf.nn.bidirectional_dynamic_rnn(cellf2, cellb2, bi_outputs_x, dtype = tf.float32, sequence_length = seq_len_x)
		bi_outputs_y_2_, state_y_2 = tf.nn.bidirectional_dynamic_rnn(cellf2, cellb2, bi_outputs_y, dtype = tf.float32, sequence_length = seq_len_y)

	output_x = tf.concat([bi_outputs_x_2_[0], bi_outputs_x_2_[1]], axis = 2)
	output_y = tf.concat([bi_outputs_y_2_[0], bi_outputs_y_2_[1]], axis = 2)

	"""
	with tf.variable_scope("layer3"):
		cellf3 = tf.nn.rnn_cell.LSTMCell(num_units=lstm_cell_size, state_is_tuple=True)
		cellb3 = tf.nn.rnn_cell.LSTMCell(num_units=lstm_cell_size, state_is_tuple=True)

		bi_outputs_x_3, state_x_3 = tf.nn.bidirectional_dynamic_rnn(cellf3, cellb3, bi_outputs_x_2, dtype = tf.float32, sequence_length = seq_len_x)
		bi_outputs_y_3, state_y_3 = tf.nn.bidirectional_dynamic_rnn(cellf3, cellb3, bi_outputs_y_2, dtype = tf.float32, sequence_length = seq_len_y)
	
	output_x = tf.concat([bi_outputs_x_[0], bi_outputs_x_[1]], axis = 2)
	output_y = tf.concat([bi_outputs_y_[0], bi_outputs_y_[1]], axis = 2)
	"""

	sum_x_temp = tf.reduce_sum(output_x, axis = 1)
	sum_x = sum_x_temp/tf.reshape(tf.cast(seq_len_x, tf.float32), (-1, 1))
	sum_x = tf.identity(sum_x, name="sum_x")
	sum_y_temp = tf.reduce_sum(output_y, axis = 1)
	sum_y = sum_y_temp/tf.reshape(tf.cast(seq_len_y, tf.float32), (-1, 1))

	output = tf.concat([sum_x, sum_y], axis = 1)
	
	"""
	w = state_x[0][1]
	x = state_x[1][1]
	y = state_y[0][1]
	z = state_y[1][1]
	output = tf.concat([w, x, y, z], axis = 1)
	"""
	W1 = tf.get_variable("W1", [ lstm_cell_size*4 , lstm_cell_size], initializer=tf.contrib.layers.xavier_initializer(seed=0))
	layer1 = tf.tensordot(output, W1, 1)

	W = tf.get_variable("W", [ lstm_cell_size , num_classes], initializer=tf.contrib.layers.xavier_initializer(seed=0))
	output_layer = tf.tensordot(layer1, W, 1)

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
	num_epochs = 20
	
	saver = tf.train.Saver()

	print(len(train_timex1))

	with tf.Session() as sess:
		tf.set_random_seed(0)
		sess.run(init)
		for i in range(0, num_epochs):
			pred_output_mat = np.zeros((num_classes, num_classes))
			print("Epoch#"+str(i) + " started")
			acc = 0
			total = 0
			acc_train = 0
			total_train = 0
			loss_this_iter = 0
			time_now = datetime.now()

			for j in range(int(len(train_timex1)/batch_size)):
				indices = random.sample(range(len(train_timex1)), batch_size)
				
				[ _, loss_this_instance, domain_predicted, output_test] = sess.run([ train_op, loss, domain_pred, output_layer], feed_dict = {
					seq_x: train_timex1[indices], seq_y: train_timex2[indices], seq_len_x : train_length1[indices], seq_len_y: train_length2[indices],
					label: train_label[indices]}) 

				loss_this_iter += loss_this_instance

			print("loss:" + repr(loss_this_iter) + "\n")


			for ex_idx in range(0, int(len(dev_timex1)/batch_size)):
				indices = range(batch_size*ex_idx, batch_size*ex_idx + batch_size)
				[predicted, output_test] = sess.run([ domain_pred, output_layer], feed_dict = {
					seq_x: dev_timex1[indices], seq_y: dev_timex2[indices], seq_len_x : dev_length1[indices], seq_len_y: dev_length2[indices]}) 

				actual = dev_label[indices]
				for idx in range(len(actual)):
					pred_output_mat[predicted[idx]][actual[idx]] += 1
					if predicted[idx] == actual[idx]:
						acc += 1
					total += 1

			for ex_idx in range(0, int(len(train_timex1)/batch_size)):
				indices = range(batch_size*ex_idx, batch_size*ex_idx + batch_size)
				[predicted] = sess.run([ domain_pred], feed_dict = {
					seq_x: train_timex1[indices], seq_y: train_timex2[indices], seq_len_x : train_length1[indices], seq_len_y: train_length2[indices]})

				actual = train_label[indices]

				for idx in range(len(actual)):
					if predicted[idx] == actual[idx]:
						acc_train += 1
					total_train += 1

			print("train accuracy for epoch: " + str(float(acc_train)/float(total_train)))
			print("test accuracy for epoch: " + str(float(acc)/float(total)))
			print(pred_output_mat)

			print_f1_scores(pred_output_mat, num_classes)

		save_path = saver.save(sess, os.path.join(output_folder, "model.ckpt"))
		print("Model saved in path: %s" % save_path)

def build_glove_model(train_exs, dev_exs, word_embeddings):

	seq_len_max = 10
	num_classes = 2
	batch_size = 16

	train_timex1 = np.asarray([np.mean(np.take(word_embeddings.vectors,word_embeddings.word_indexer.get_index_list(ex.timex1_tokens), axis = 0), axis = 0) for ex in train_exs])
	train_timex2 = np.asarray([np.mean(np.take(word_embeddings.vectors,word_embeddings.word_indexer.get_index_list(ex.timex2_tokens), axis = 0), axis = 0) for ex in train_exs])
	train_label = np.array([ex.label for ex in train_exs])

	dev_timex1 = np.asarray([np.mean(np.take(word_embeddings.vectors,word_embeddings.word_indexer.get_index_list(ex.timex1_tokens), axis = 0), axis = 0) for ex in dev_exs])
	dev_timex2 = np.asarray([np.mean(np.take(word_embeddings.vectors,word_embeddings.word_indexer.get_index_list(ex.timex2_tokens), axis = 0), axis = 0) for ex in dev_exs])
	dev_label = np.array([ex.label for ex in dev_exs])

	print(np.sum(train_label))
	print(len(train_label))

	timex1 = tf.placeholder(tf.float32, [batch_size , 300], name= "p1")
	timex2 = tf.placeholder(tf.float32, [batch_size , 300], name = "p2")
	label = tf.placeholder(tf.int32, [batch_size], name = "label")

	input_ = tf.concat([timex1, timex2], axis = 1)


	W1 = tf.get_variable("W1", [ 600 , 50], initializer=tf.contrib.layers.xavier_initializer(seed=0))
	layer = tf.tensordot(input_, W1, 1)

	W = tf.get_variable("W", [ 50 , num_classes], initializer=tf.contrib.layers.xavier_initializer(seed=0))
	output_layer = tf.tensordot(layer, W, 1)

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
	num_epochs = 30


	with tf.Session() as sess:
		tf.set_random_seed(0)
		sess.run(init)
		for i in range(0, num_epochs):
			pred_output_mat = np.zeros((num_classes, num_classes))
			print("Epoch#"+str(i) + " started")
			acc = 0
			total = 0
			acc_train = 0
			total_train = 0
			loss_this_iter = 0
			time_now = datetime.now()

			for j in range(int(len(train_timex1)/batch_size)):
				indices = random.sample(range(len(train_timex1)), batch_size)
				
				[ _, loss_this_instance, domain_predicted, output_test] = sess.run([ train_op, loss, domain_pred, output_layer], feed_dict = {
					timex1: train_timex1[indices], timex2: train_timex2[indices], label: train_label[indices]}) 

				loss_this_iter += loss_this_instance

			print("loss:" + repr(loss_this_iter) + "\n")


			for ex_idx in range(0, int(len(dev_timex1)/batch_size)):
				indices = range(batch_size*ex_idx, batch_size*ex_idx + batch_size)
				[predicted, output_test] = sess.run([ domain_pred, output_layer], feed_dict = {
					timex1: dev_timex1[indices], timex2: dev_timex2[indices], label: dev_label[indices]})  

				actual = dev_label[indices]

				for idx in range(len(actual)):
					pred_output_mat[predicted[idx]][actual[idx]] += 1
					if predicted[idx] == actual[idx]:
						acc += 1
					total += 1

			for ex_idx in range(0, int(len(train_timex1)/batch_size)):
				indices = range(batch_size*ex_idx, batch_size*ex_idx + batch_size)
				[predicted] = sess.run([ domain_pred], feed_dict = {
					timex1: train_timex1[indices], timex2: train_timex2[indices], label: train_label[indices]}) 

				actual = train_label[indices]

				for idx in range(len(actual)):
					if predicted[idx] == actual[idx]:
						acc_train += 1
					total_train += 1

			print("train accuracy for epoch: " + str(float(acc_train)/float(total_train)))
			print("test accuracy for epoch: " + str(float(acc)/float(total)))
			print(pred_output_mat)
			print("\n")


def build_elmo_model(train_exs, dev_exs):
	seq_len_max = 10
	num_classes = 2
	batch_size = 16

	print(len(train_exs))

	train_timex1 = np.array([ex.elmo1 for ex in train_exs])
	train_timex2 = np.array([ex.elmo2 for ex in train_exs])
	train_label = np.array([ex.label for ex in train_exs])

	print(train_label)

	dev_timex1 = np.array([ex.elmo1 for ex in dev_exs])
	dev_timex2 = np.array([ex.elmo2 for ex in dev_exs])
	dev_label = np.array([ex.label for ex in dev_exs])

	timex1 = tf.placeholder(tf.float32, [batch_size , train_timex1.shape[1]], name= "p1")
	timex2 = tf.placeholder(tf.float32, [batch_size , train_timex1.shape[1]], name = "p2")
	label = tf.placeholder(tf.int32, [batch_size], name = "label")

	input_ = tf.subtract(timex1, timex2)

	W1 = tf.get_variable("W1", [ input_.shape[1] , num_classes], initializer=tf.contrib.layers.xavier_initializer(seed=0))
	layer1 = tf.tensordot(input_, W1, 1)
	"""
	W3 = tf.get_variable("W3", [ 2000 , 1000], initializer=tf.contrib.layers.xavier_initializer(seed=0))
	layer2 = tf.tensordot(layer1, W3, 1)

	W4 = tf.get_variable("W4", [ 1000 , 500], initializer=tf.contrib.layers.xavier_initializer(seed=0))
	layer3 = tf.tensordot(layer2, W4, 1)

	W5 = tf.get_variable("W5", [ 500 , 50], initializer=tf.contrib.layers.xavier_initializer(seed=0))
	layer4 = tf.tensordot(layer3, W5, 1)

	W = tf.get_variable("W", [ 50 , num_classes], initializer=tf.contrib.layers.xavier_initializer(seed=0))
	output_layer = tf.tensordot(layer4, W, 1)
	
	output_layer = tf.identity(output_layer, name="output_layer")
	"""
	loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = layer1, labels = label)
	loss = tf.reduce_mean(loss)

	domain_pred = tf.argmax(layer1,axis=1)
	domain_pred = tf.identity(domain_pred, name="domain_pred")


	decay_steps = 10
	learning_rate_decay_factor = .999995
	global_step = tf.contrib.framework.get_or_create_global_step()
	initial_learning_rate = 0.0001


	lr = tf.train.exponential_decay(initial_learning_rate, global_step, decay_steps, learning_rate_decay_factor, staircase=True)    
	optimizer = tf.train.AdamOptimizer(lr)
	grads = optimizer.compute_gradients(loss)
	apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)
	with tf.control_dependencies([apply_gradient_op]):
		train_op = tf.no_op(name='train')


	init = tf.global_variables_initializer()
	num_epochs = 10
	
	saver = tf.train.Saver()

	with tf.Session() as sess:
		tf.set_random_seed(0)
		sess.run(init)
		for i in range(0, num_epochs):
			pred_output_mat = np.zeros((num_classes, num_classes))
			print("Epoch#"+str(i) + " started")
			acc = 0
			total = 0
			acc_train = 0
			total_train = 0
			loss_this_iter = 0
			time_now = datetime.now()

			for j in range(int(len(train_timex1)/batch_size)):
				indices = random.sample(range(len(train_timex1)), batch_size)
				
				[ _, loss_this_instance, domain_predicted, output_test] = sess.run([ train_op, loss, domain_pred, output_layer], feed_dict = {
					timex1: train_timex1[indices], timex2: train_timex2[indices], label: train_label[indices]}) 

				loss_this_iter += loss_this_instance

			print("loss:" + repr(loss_this_iter) + "\n")


			for ex_idx in range(0, int(len(dev_timex1)/batch_size)):
				indices = range(batch_size*ex_idx, batch_size*ex_idx + batch_size)
				[predicted, output_test] = sess.run([ domain_pred, output_layer], feed_dict = {
					timex1: dev_timex1[indices], timex2: dev_timex2[indices], label: dev_label[indices]})  

				actual = dev_label[indices]

				for idx in range(len(actual)):
					pred_output_mat[predicted[idx]][actual[idx]] += 1
					if predicted[idx] == actual[idx]:
						acc += 1
					total += 1

			for ex_idx in range(0, int(len(train_timex1)/batch_size)):
				indices = range(batch_size*ex_idx, batch_size*ex_idx + batch_size)
				[predicted] = sess.run([ domain_pred], feed_dict = {
					timex1: train_timex1[indices], timex2: train_timex2[indices], label: train_label[indices]}) 

				actual = train_label[indices]

				for idx in range(len(actual)):
					if predicted[idx] == actual[idx]:
						acc_train += 1
					total_train += 1

			print("train accuracy for epoch: " + str(float(acc_train)/float(total_train)))
			print("test accuracy for epoch: " + str(float(acc)/float(total)))
			print(pred_output_mat)

			print_f1_scores(pred_output_mat, num_classes)

		save_path = saver.save(sess, os.path.join(output_folder, "model.ckpt"))
		print("Model saved in path: %s" % save_path)

def pad_to_length_2d(input_, max_len):
	output = np.zeros((max_len, input_.shape[1]))
	for i in range(input_.shape[0]):
		output[i] = input_[i]

	return output

def build_elmo_model_seq(train_exs, dev_exs, output_folder):
	max_len = 10

	num_classes = 6
	batch_size = 16
	lstm_cell_size = 25

	train_timex1 = np.array([pad_to_length_2d(ex.elmo1, max_len) for ex in train_exs])
	train_timex2 = np.array([pad_to_length_2d(ex.elmo2, max_len) for ex in train_exs])
	train_length1 = np.array([len(ex.elmo1) for ex in train_exs])
	train_length2 = np.array([len(ex.elmo2) for ex in train_exs])
	train_label = np.array([ex.label for ex in train_exs])

	dev_timex1 = np.array([pad_to_length_2d(ex.elmo1, max_len) for ex in dev_exs])
	dev_timex2 = np.array([pad_to_length_2d(ex.elmo2, max_len) for ex in dev_exs])
	dev_length1 = np.array([len(ex.elmo1) for ex in dev_exs])
	dev_length2 = np.array([len(ex.elmo2) for ex in dev_exs])
	dev_label = np.array([ex.label for ex in dev_exs])	


	seq_x = tf.placeholder(tf.float32, [batch_size ,max_len, train_timex1[0].shape[1]], name= "p1")
	seq_y = tf.placeholder(tf.float32, [batch_size ,max_len, train_timex1[0].shape[1]], name = "p2")
	seq_len_x = tf.placeholder(tf.int32, batch_size, name = "p3")
	seq_len_y = tf.placeholder(tf.int32, batch_size, name = "p4")
	label = tf.placeholder(tf.int32, [batch_size], name = "label")

	W3 = tf.get_variable("W3", [ seq_x.shape[2] , 1000], initializer=tf.contrib.layers.xavier_initializer(seed=0))
	W4 = tf.get_variable("W4", [ 1000 , 100], initializer=tf.contrib.layers.xavier_initializer(seed=0))
	
	seq_x_1 = tf.tensordot(seq_x, W3, 1)
	seq_x_2 = tf.tensordot(seq_x_1, W4, 1)

	seq_y_1 = tf.tensordot(seq_y, W3, 1)
	seq_y_2 = tf.tensordot(seq_y_1, W4, 1)


	cellf1 = tf.nn.rnn_cell.LSTMCell(num_units=lstm_cell_size, state_is_tuple=True)
	cellb1 = tf.nn.rnn_cell.LSTMCell(num_units=lstm_cell_size, state_is_tuple=True)

	bi_outputs_x_, state_x = tf.nn.bidirectional_dynamic_rnn(cellf1, cellb1, seq_x_2, dtype = tf.float32, sequence_length = seq_len_x)
	bi_outputs_y_, state_y = tf.nn.bidirectional_dynamic_rnn(cellf1, cellb1, seq_y_2, dtype = tf.float32, sequence_length = seq_len_y)

	
	bi_outputs_x = tf.concat([bi_outputs_x_[0], bi_outputs_x_[1]], axis = 2)
	bi_outputs_y = tf.concat([bi_outputs_y_[0], bi_outputs_y_[1]], axis = 2)

	sum_x_temp = tf.reduce_sum(bi_outputs_x, axis = 1)
	sum_x = sum_x_temp/tf.reshape(tf.cast(seq_len_x, tf.float32), (-1, 1))
	sum_x = tf.identity(sum_x, name="sum_x")
	sum_y_temp = tf.reduce_sum(bi_outputs_y, axis = 1)
	sum_y = sum_y_temp/tf.reshape(tf.cast(seq_len_y, tf.float32), (-1, 1))

	output = tf.concat([sum_x, sum_y], axis = 1)

	W1 = tf.get_variable("W1", [ lstm_cell_size*4 , lstm_cell_size], initializer=tf.contrib.layers.xavier_initializer(seed=0))
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
	initial_learning_rate = 0.0001


	lr = tf.train.exponential_decay(initial_learning_rate, global_step, decay_steps, learning_rate_decay_factor, staircase=True)    
	optimizer = tf.train.AdamOptimizer(lr)
	grads = optimizer.compute_gradients(loss)
	apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)
	with tf.control_dependencies([apply_gradient_op]):
		train_op = tf.no_op(name='train')


	init = tf.global_variables_initializer()
	num_epochs = 10
	
	saver = tf.train.Saver()

	with tf.Session() as sess:
		tf.set_random_seed(0)
		sess.run(init)
		for i in range(0, num_epochs):
			pred_output_mat = np.zeros((num_classes, num_classes))
			print("Epoch#"+str(i) + " started")
			acc = 0
			total = 0
			acc_train = 0
			total_train = 0
			loss_this_iter = 0
			time_now = datetime.now()

			for j in range(int(len(train_timex1)/batch_size)):
				indices = random.sample(range(len(train_timex1)), batch_size)
				
				[ _, loss_this_instance, domain_predicted, output_test] = sess.run([ train_op, loss, domain_pred, output_layer], feed_dict = {
					seq_x: train_timex1[indices], seq_y: train_timex2[indices], seq_len_x : train_length1[indices], seq_len_y: train_length2[indices],
					label: train_label[indices]}) 

				loss_this_iter += loss_this_instance

			print("loss:" + repr(loss_this_iter) + "\n")


			for ex_idx in range(0, int(len(dev_timex1)/batch_size)):
				indices = range(batch_size*ex_idx, batch_size*ex_idx + batch_size)
				[predicted, output_test] = sess.run([ domain_pred, output_layer], feed_dict = {
					seq_x: dev_timex1[indices], seq_y: dev_timex2[indices], seq_len_x : dev_length1[indices], seq_len_y: dev_length2[indices]}) 

				actual = dev_label[indices]

				for idx in range(len(actual)):
					pred_output_mat[predicted[idx]][actual[idx]] += 1
					if predicted[idx] == actual[idx]:
						acc += 1
					total += 1

			for ex_idx in range(0, int(len(train_timex1)/batch_size)):
				indices = range(batch_size*ex_idx, batch_size*ex_idx + batch_size)
				[predicted] = sess.run([ domain_pred], feed_dict = {
					seq_x: train_timex1[indices], seq_y: train_timex2[indices], seq_len_x : train_length1[indices], seq_len_y: train_length2[indices]})

				actual = train_label[indices]

				for idx in range(len(actual)):
					if predicted[idx] == actual[idx]:
						acc_train += 1
					total_train += 1

			print("train accuracy for epoch: " + str(float(acc_train)/float(total_train)))
			print("test accuracy for epoch: " + str(float(acc)/float(total)))
			print(pred_output_mat)

			print_f1_scores(pred_output_mat, num_classes)

		save_path = saver.save(sess, os.path.join(output_folder, "model.ckpt"))
		print("Model saved in path: %s" % save_path)











