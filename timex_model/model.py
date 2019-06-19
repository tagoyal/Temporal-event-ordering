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
	num_classes = 3
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


	sum_x_temp = tf.reduce_sum(output_x, axis = 1)
	sum_x = sum_x_temp/tf.reshape(tf.cast(seq_len_x, tf.float32), (-1, 1))
	sum_x = tf.identity(sum_x, name="sum_x")
	sum_y_temp = tf.reduce_sum(output_y, axis = 1)
	sum_y = sum_y_temp/tf.reshape(tf.cast(seq_len_y, tf.float32), (-1, 1))

	output = tf.concat([sum_x, sum_y], axis = 1)
	
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
			print("dev accuracy for epoch: " + str(float(acc)/float(total)))

		save_path = saver.save(sess, os.path.join(output_folder, "model.ckpt"))
		print("Model saved in path: %s" % save_path)

