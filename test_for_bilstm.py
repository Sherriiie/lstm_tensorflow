# -*- coding: UTF-8 -*-

import tensorflow as tf
import numpy as np
import data_processor

# define a LSTM module
num_nodes = 141 # 141      # number of hidden layer units
embedding_size = 100     # 100
batch_size = 500        #100, margin =0.01
seq_len = 200
loss_margin = 0.1
learning_rate = 0.1
ratio = batch_size     # for test  == batch_size
test_size = 100 #100

# ==============================================================================
# reference: LSTM-based deep learning models for non-factoid answer selection
#
# ==============================================================================
# Build vocabulary first
print("------ test for bilstm, Loading data ------")
filePath = '/home/sherrie/PycharmProjects/tensorflow/lstm/'
vocab = data_processor.buildVocab(filePath)
vocab_size = len(vocab)


graph = tf.Graph()
with graph.as_default():
	# Parameters of input gate, forget gate, memory cell and output gate
	# Input gate: input, previous output, and bias.
	ix = tf.Variable(tf.truncated_normal([embedding_size, num_nodes], -0.1, 0.1))
	im = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
	ib = tf.Variable(tf.zeros([1, num_nodes]))
	# Forget gate: input, previous output, and bias.
	fx = tf.Variable(tf.truncated_normal([embedding_size, num_nodes], -0.1, 0.1))
	fm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
	fb = tf.Variable(tf.zeros([1, num_nodes]))
	# Memory cell: input, state and bias.
	cx = tf.Variable(tf.truncated_normal([embedding_size, num_nodes], -0.1, 0.1))
	cm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
	cb = tf.Variable(tf.zeros([1, num_nodes]))
	# Output gate: input, previous output, and bias.
	ox = tf.Variable(tf.truncated_normal([embedding_size, num_nodes], -0.1, 0.1))
	om = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
	ob = tf.Variable(tf.zeros([1, num_nodes]))
	# Variables saving state across unrollings.
	#saved_output = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
	#saved_state = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
	saved_output = tf.zeros([batch_size, num_nodes])
	saved_state = tf.zeros([batch_size, num_nodes])


	# Definition of the cell computation.
	def lstm_cell(i, o, state):
		"""Create a LSTM cell. See e.g.: http://arxiv.org/pdf/1402.1128v1.pdf
		Note that in this formulation, we omit the various connections between the
		previous state and the gates."""
		input_gate = tf.sigmoid(tf.matmul(i, ix) + tf.matmul(o, im) + ib)
		# print('input_gate', (tf.matmul(i, ix)).get_shape(), (tf.matmul(o, im)).get_shape(), (ib).get_shape())
		forget_gate = tf.sigmoid(tf.matmul(i, fx) + tf.matmul(o, fm) + fb)
		update = tf.matmul(i, cx) + tf.matmul(o, cm) + cb  # transform of input
		state = forget_gate * state + input_gate * tf.tanh(update)
		'''
		print ('**** i', i.get_shape())
		print ('**** ox', ox.get_shape())
		print ('**** ', o.get_shape(), om.get_shape(), ob.get_shape())'''
		output_gate = tf.sigmoid(tf.matmul(i, ox) + tf.matmul(o, om) + ob)
		output = output_gate * tf.tanh(state)
		return output, state


	def bilstm(train_inputs):
		train_inputs = tf.unpack(train_inputs,axis=0)
		outputs_f = list()
		outputs_b = list()
		output = saved_output
		state = saved_state
		# print(train_inputs.get_shape())
		for i in train_inputs:
			output, state = lstm_cell(i, output, state)
			outputs_f.append(output)  # sequence_length*batch_size*node_num
		print('outputs_forward', np.shape(outputs_f), outputs_f[0].get_shape())

		output = saved_output
		state = saved_state
		# print(train_inputs.get_shape())
		for i in range(len(train_inputs)):
			output, state = lstm_cell(train_inputs[len(train_inputs) - i - 1], output, state)
			outputs_b.append(output)  # sequence_length*batch_size*node_num

		# concatenate the forward outputs and backward outputs　for every word
		output_seq = list()
		for i in range(seq_len):
			output_seq.append( tf.concat(1, [outputs_b[i], outputs_f[seq_len - i-1]]) )
		print('length of outputs_seq', len(output_seq), ', shape of the elements', output_seq[0].get_shape())
		return output_seq


	# max-pooling for the input, which is a list of tensors
	def max_pooling(input):
		att_input = tf.pack(input, axis=1)  # (batch_size, seq_len, 2*num_nodes)
		att_input_expand = tf.expand_dims(att_input,-1)  # expand dims to match the max-pooling tensors' requisories(batch_size, height, width, channels).
		att_pool = tf.nn.max_pool(att_input_expand,
									ksize=[1, seq_len, 1, 1],
									strides=[1, 1, 1, 1],
									padding='VALID',
									name="pool")  # shape of pooled is [batch_size,1,1,1]
		att_out = tf.reshape(att_pool, [-1, 2 * num_nodes])  # (batch_size, 2*num_nodes)
		return att_out


	def embedding(inputs):
		# embedding layer, embed the words into vectors and expand them to 4 dim tensors for cnn architecture.
		with tf.device('/cpu:0'), tf.name_scope("embedding"):
			W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
			                name="W")  # W is embedding matrix
			embedded_words= tf.nn.embedding_lookup(W, inputs)       # seq_len*batch_size*embedding_size
			# transpose it to satify requirments of bilstm
			# embedded_words = tf.transpose(embedded_words1, [1,0,2])
		return embedded_words


	# Input data. To be fed in with dictionary.
	train_inputs_x1 = list()        # question
	train_inputs_x2 = list()        # positive answer
	train_inputs_x3 = list()        # negtive answer

	for _ in range(seq_len):
		train_inputs_x1.append(
			tf.placeholder(tf.int32, shape=[batch_size]))     # seq_len*batch_size
		train_inputs_x2.append(
			tf.placeholder(tf.int32, shape=[batch_size]))
		train_inputs_x3.append(
			tf.placeholder(tf.int32, shape=[batch_size]))
	print ('train_inputs_x1', len(train_inputs_x1),train_inputs_x1[0].get_shape())
	test = tf.pack(train_inputs_x1)
	print ('test', test.get_shape())

	train_inputs_q = embedding(tf.pack(train_inputs_x1))        #seq_len*batch_size*embedding_size
	train_inputs_ap = embedding(tf.pack(train_inputs_x2))
	train_inputs_an = embedding(tf.pack(train_inputs_x3))

	# the output of biLSTM for three kinds of inputs
	output_seq_q = bilstm(train_inputs_q)
	output_seq_ap = bilstm(train_inputs_ap)
	output_seq_an = bilstm(train_inputs_an)
	# do max-pooling for the question-outputs
	output_q = max_pooling(output_seq_q)
	output_ap = max_pooling(output_seq_ap)
	output_an = max_pooling(output_seq_an)

	def cal_length(var):
		len_var = tf.sqrt(tf.reduce_sum(tf.mul(var, var),1))
		return len_var


	def cal_similarity(question, answer):
		len_ques = cal_length(question)
		len_ans= cal_length(answer)
		multi = tf.reduce_sum(tf.mul(question, answer), 1)
		cos= tf.div(multi, tf.mul(len_ques, len_ans))
		return cos

	cos_pos = cal_similarity(output_q, output_ap)
	cos_neg = cal_similarity(output_q, output_an)
	print ('cosine', cos_pos.get_shape(), cos_neg.get_shape())

	zero = tf.constant(0, shape=[batch_size], dtype=tf.float32)
	margin = tf.constant(loss_margin, shape=[batch_size], dtype=tf.float32)

	with tf.name_scope("loss"):
		losses = tf.maximum(zero, tf.sub(margin, tf.sub(cos_pos, cos_neg)))
		loss = tf.reduce_sum(losses)

	# accuracy
	with tf.name_scope("accuracy"):
		correct = tf.equal(zero, losses)
		accuracy = tf.reduce_mean(tf.cast(correct, "float"), name="accuracy")


		# define a validation/test step
	def test_step(input_y1, input_y2, input_y3, sess):
		feed_dict = dict()
		feed_y1 = np.transpose(input_y1, [1, 0])    # seq_len*ratio
		feed_y2 = np.transpose(input_y2, [1, 0])    # seq_len*ratio
		feed_y3 = np.transpose(input_y3, [1, 0])    # seq_len*ratio
		# print ('feed_x1', np.shape(feed_x1))

		for i in range(seq_len):
			feed_dict[train_inputs_x1[i]] = feed_y1[i]
			feed_dict[train_inputs_x2[i]] = feed_y2[i]
			feed_dict[train_inputs_x3[i]] = feed_y3[i]

		correct_flag = 0
		test_loss_ = sess.run([loss], feed_dict)
		# print('The loss of validation is {}'.format(loss))
		if test_loss_ == [0.0]:
			correct_flag = 1
		cos_pos_, cos_neg_, accuracy_ = sess.run([cos_pos, cos_neg, accuracy], feed_dict)
		output_q_, output_ap_, output_an_ = sess.run([output_q, output_ap, output_an], feed_dict)
		# data_processor.saveFeatures(output_q_, output_ap_, 0 , 0)
		# data_processor.saveFeatures(output_q_, output_an_, 0, 0)
		data_processor.saveFeatures(cos_pos_, cos_neg_, test_loss_, accuracy_)
		return correct_flag


	# 显示/保存测试数据
	def save_test_data(y1, y2, y3, i):
		sen_y1 = data_processor.getSentence(y1, vocab)[0]
		sen_y2 = data_processor.getSentence(y2, vocab)[0]
		sen_y3 = data_processor.getSentence(y3, vocab)
		data_processor.saveData('\nQuestion ' + str(i + 1) + ':\n' + sen_y1)
		data_processor.saveData('\nPositive Answer:\n' + sen_y2)
		data_processor.saveData('\nNegative Answers:')
		for j in range(4):        # to do
			data_processor.saveData('\n' + str(j + 1) + ' ' + sen_y3[j])
		return


	def test():
		correct_num = int(0)
		for i in range(test_size):
			batch_y1, batch_y2, batch_y3 = data_processor.loadValData(vocab, filePath, seq_len, ratio)      # batch_size*seq_len
			# 显示/保存测试数据

			save_test_data(batch_y1, batch_y2, batch_y3, i)
			correct_flag = test_step(batch_y1, batch_y2, batch_y3,sess)
			# print ('corr_flag', correct_flag)
			correct_num += correct_flag
		print ('correct_num',correct_num)
		acc = correct_num / float(test_size)
		return acc


	f = open(filePath + "data/saved_features.txt", 'w')
	f.close()
	f = open(filePath + "data/saved_test_data.txt", 'w')
	f.close()



	# Launch the graph
	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())
		saver = tf.train.Saver(tf.all_variables())
		saver.restore(sess,
		              '/home/sherrie/PycharmProjects/tensorflow/lstm/checkpoints_train/bilstm/step200_train_acc0.98_test_acc0.1')

		print('\n============================>restored,  begin to test ')

		acc = test()
		print(
			'--------The test result among the test data sets: acc = {}, test size = {}, test ratio = {}----------'.format(
				acc, test_size, ratio))