import tensorflow as tf
import os
from matplotlib import pyplot as plt
import data
import cv2
import numpy as np

class Common:
	def __init__(self, model_name, input_patch_size, lr_value, lr_decay_rate, lr_decay_freq, m_value, batch_size):
		self.model_name = model_name
		self.input_patch_size = input_patch_size
		self.lr_value = lr_value / lr_decay_rate
		self.lr_decay_rate = lr_decay_rate
		self.lr_decay_freq = lr_decay_freq
		self.m_value = m_value
		self.batch_size = batch_size

		self.x_image = tf.placeholder('float', shape=[None, input_patch_size, input_patch_size, 3])
		self.phase_train = tf.Variable(True)
		self.lr = tf.placeholder(tf.float32)
		self.m = tf.placeholder(tf.float32)
		self.keep_prob = tf.placeholder(tf.float32)

		self.g = tf.Graph()

	def weight_variable(self, shape, seed):
		initial = tf.truncated_normal(shape, stddev=0.1, seed=seed)
		return tf.Variable(initial)

	def bias_variable(self, shape):
		initial = tf.constant(0.1, shape=shape)
		return tf.Variable(initial)

	def conv2d(self, x, W):
		return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

	def conv2d_stride(self, x, W):
		return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')

	def conv2d_stride4(self, x, W):
		   return tf.nn.conv2d(x, W, strides=[1, 4, 4, 1], padding='SAME')

	def max_pool_2x2(self, x):
		return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')

	def max_pool_2x2_stride(self, x):
		return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

	def batch_norm(self, x, n_out, phase_train):
		"""
		Batch normalization on convolutional maps.
		Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
		Args:
			x:		   Tensor, 4D BHWD input maps
			n_out:	   integer, depth of input maps
			phase_train: boolean tf.Varialbe, true indicates training phase
			scope:	   string, variable scope
		Return:
			normed:	  batch-normalized maps
		"""
		with tf.variable_scope('bn'):
			beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
										 name='beta', trainable=True)
			gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
										  name='gamma', trainable=True)
			batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
			ema = tf.train.ExponentialMovingAverage(decay=0.5)

			def mean_var_with_update():
				ema_apply_op = ema.apply([batch_mean, batch_var])
				with tf.control_dependencies([ema_apply_op]):
					return tf.identity(batch_mean), tf.identity(batch_var)

			mean, var = tf.cond(phase_train,
								mean_var_with_update,
								lambda: (ema.average(batch_mean), ema.average(batch_var)))
			normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
		return normed

	def restore(self):
		saver = tf.train.Saver()

		sess = tf.Session()
		saver.restore(sess, "trained_model/%s/Drone_CNN.ckpt" % self.model_name)
		print("Model restored.")
		return sess

class Common_label(Common):
	def __init__(self, model_name, input_patch_size, lr_value, lr_decay_rate, lr_decay_freq, m_value, batch_size):
		Common.__init__(self, model_name, input_patch_size, lr_value, lr_decay_rate, lr_decay_freq, m_value, batch_size)
		self.y_ = tf.placeholder('float', shape=[None, 3])

	def train(self, epoch):
		os.makedirs('tb/%s' % self.model_name)
		os.makedirs('trained_model/%s' % self.model_name)

		tf.summary.image('input x_image', self.x_image)
		tf.summary.scalar('train_accuracy', self.accuracy)
		tf.summary.scalar('cross_entropy', self.cross_entropy)
		tf.summary.scalar('learning rate', self.lr)

		sess = tf.Session()

		sess.run(tf.global_variables_initializer())
		sess.run(tf.local_variables_initializer())

		saver = tf.train.Saver()

		f_log = open('trained_model/%s/log.csv' % self.model_name, 'w')

		merged = tf.summary.merge_all()
		train_writer = tf.summary.FileWriter('/home/lsmjn/Drone-CNN/tb/%s' % self.model_name, sess.graph)

		ngii_dir_training = data.get_ngii_dir('training')
		ngii_dir_test = data.get_ngii_dir('test')

		conn, cur = data.get_db_connection()

		steps = data.get_steps(self.batch_size)

		k = 0

		print('\nCurrent Model: %s' % self.model_name)

		for i in range(0, epoch):
			for j in range(0, steps):
				x_batch, y_batch, _ = data.make_batch(conn, cur, 'training', self.batch_size)

				if k%10 == 0:
					print('\nstep %d, epoch %d' % (k, i))
					train_xe, train_accuracy = sess.run([self.cross_entropy, self.accuracy], feed_dict={self.x_image:x_batch, self.y_:y_batch, self.keep_prob: 1.0})
					print('Train XE:')
					print(train_xe)
					print('Train Accuracy:')
					print(train_accuracy)

					for l in range(0, len(ngii_dir_test)):
						dataset_test_name = ngii_dir_test[l][0]
						x_batch_test, y_batch_test, _ = data.make_batch(conn, cur, 'test', self.batch_size)
						test_accuracy = sess.run(self.accuracy, feed_dict={self.x_image:x_batch_test, self.y_:y_batch_test, self.keep_prob: 1.0})
						print('Test Accuracy:')
						print(test_accuracy)

					f_log.write('%d,%d,%f,%f,%f\n' % (i, k, train_xe, train_accuracy, test_accuracy))
				if k%self.lr_decay_freq == 0:
					self.lr_value = self.lr_value * 0.1
					print('Learning rate:')
					print(self.lr_value)

				summary, _ = sess.run([merged, self.train_step], feed_dict={self.x_image: x_batch, self.y_: y_batch, self.lr:self.lr_value, self.m:self.m_value, self.keep_prob: 0.5})
				#sess.run(self.train_step, feed_dict={self.x_image: x_batch, self.y_: y_batch, self.lr:self.lr_value, self.m:self.m_value, self.keep_prob: 0.5})
				train_writer.add_summary(summary, k)
				k = k + 1

		cur.close()
		conn.close()

		save_path = saver.save(sess, "trained_model/%s/Drone_CNN.ckpt" % self.model_name)
		print('Model saved in file: %s' % save_path)
		#train_writer.close()
		f_log.close()

	def create_test_patches(self):
		y_conv_argmax = tf.argmax(self.y_conv, 1)
		with self.restore() as sess:
			conn, cur = data.get_db_connection()
			x_batch_test, y_batch_test, _ = data.make_batch(conn, cur, 'test', 20)
			y_prediction, test_accuracy = sess.run([y_conv_argmax, self.accuracy], feed_dict={self.x_image:x_batch_test, self.y_:y_batch_test, self.keep_prob: 1.0})

			f, axarr = plt.subplots(2, 10)

			f.suptitle('Test Result of Test Dataset (Accuracy: %f)' % test_accuracy)

			for i in range(0, 20):
				if y_prediction[i] == 0:
					y_label = 'Building'
				elif y_prediction[i] == 1:
					y_label = 'Road'
				elif y_prediction[i] == 2:
					y_label = 'Otherwise'
				else:
					y_label = "???"
				axarr[0, i].imshow(x_batch_test[i]) if i < 10 else axarr[1, i-10].imshow(x_batch_test[i])
				axarr[0, i].set_title(y_label) if i < 10 else axarr[1, i-10].set_title(y_label)

			plt.show()

	def run_prediction(self, sess, x_batch_drone, y_conv_argmax, y_conv_softmax, mode, interest_label, prob_list):
		if mode == 'MOST_PROBABLE_CLASS':
			pred_result = sess.run(y_conv_argmax, feed_dict={self.x_image:x_batch_drone, self.keep_prob: 1.0})
			for prob in pred_result:
				prob_list.append(prob)
		elif mode == 'PROB_OF_INTEREST':
			if interest_label == 'Building':
				interest_ch = 0
			elif interest_label == 'Road':
				interest_ch = 1
			elif interest_label == 'Otherwise':
				interest_ch = 2
			pred_result = sess.run(y_conv_softmax, feed_dict={self.x_image:x_batch_drone, self.keep_prob: 1.0})
			for prob in pred_result:
				prob_list.append(prob[interest_ch])


	def drone_prediction(self, mode, window_sliding_stride=8, interest_label='Building'):
		conn, cur = data.get_db_connection()
		drone_dir = data.get_drone_dir_all()

		y_conv_argmax = tf.argmax(self.y_conv, 1)
		y_conv_softmax= tf.nn.softmax(self.y_conv)

		with self.restore() as sess:
			#for each drone ortho-images
			for row in drone_dir:
				print('Current Dataset: %s (%s)' % (row[0], mode))
				curr_image = cv2.imread(row[1])

				x_batch_drone = []

				k = 0
				prob_list = []

				curr_image_h = len(curr_image)
				curr_image_w = len(curr_image[0])

				result_image_h = curr_image_h - curr_image_h%window_sliding_stride - self.input_patch_size
				result_image_w = curr_image_w - curr_image_w%window_sliding_stride - self.input_patch_size

				#For each patch...
				for i in range(0, result_image_h, window_sliding_stride):
					for j in range(0, result_image_w, window_sliding_stride):
						k = k + 1

						patch = np.array(curr_image[i:i+self.input_patch_size, j:j+self.input_patch_size])

						x_batch_drone.append(patch)

						#Run tensorflow with 128 patches
						if k%self.batch_size==0:
							self.run_prediction(sess, x_batch_drone, y_conv_argmax, y_conv_softmax, mode, interest_label, prob_list)
							x_batch_drone = []

				#Run prediction for rest of data
				self.run_prediction(sess, x_batch_drone, y_conv_argmax, y_conv_softmax, mode, interest_label, prob_list)

				result = np.reshape(prob_list, (int(result_image_h/window_sliding_stride), int(result_image_w/window_sliding_stride)))

				if mode == 'PROB_OF_INTEREST':
					result = cv2.resize(result, (len(curr_image[0]), len(curr_image)), interpolation=cv2.INTER_LINEAR) * 255

				cv2.imwrite('result_%s_%s.png' % (row[0], mode), result)
				print('Prediction Complete.')

class Common_single_label(Common):
	def __init__(self, model_name, input_patch_size, lr_value, lr_decay_rate, lr_decay_freq, m_value, batch_size):
		Common.__init__(self, model_name, input_patch_size, lr_value, lr_decay_rate, lr_decay_freq, m_value, batch_size)
		self.y_ = tf.placeholder('float', shape=[None, 2])

	def train(self, epoch):
		os.makedirs('tb/%s' % self.model_name)
		os.makedirs('trained_model/%s' % self.model_name)

		tf.summary.image('input x_image', self.x_image)
		tf.summary.scalar('train_accuracy', self.accuracy)
		tf.summary.scalar('cross_entropy', self.cross_entropy)
		tf.summary.scalar('learning rate', self.lr)

		sess = tf.Session()

		sess.run(tf.global_variables_initializer())
		sess.run(tf.local_variables_initializer())

		saver = tf.train.Saver()

		f_log = open('trained_model/%s/log.csv' % self.model_name, 'w')

		merged = tf.summary.merge_all()
		train_writer = tf.summary.FileWriter('/home/lsmjn/Drone-CNN/tb/%s' % self.model_name, sess.graph)

		ngii_dir_training = data.get_ngii_dir('training')
		ngii_dir_test = data.get_ngii_dir('test')

		conn, cur = data.get_db_connection()

		steps = data.get_steps(self.batch_size)

		k = 0

		print('\nCurrent Model: %s' % self.model_name)

		for i in range(0, epoch):
			for j in range(0, steps):
				x_batch, y_batch, _ = data.make_batch(conn, cur, 'training', self.batch_size, 'Building')

				if k%10 == 0:
					print('\nstep %d, epoch %d' % (k, i))
					train_xe, train_accuracy = sess.run([self.cross_entropy, self.accuracy], feed_dict={self.x_image:x_batch, self.y_:y_batch, self.keep_prob: 1.0})
					print('Train XE:')
					print(train_xe)
					print('Train Accuracy:')
					print(train_accuracy)

					for l in range(0, len(ngii_dir_test)):
						dataset_test_name = ngii_dir_test[l][0]
						x_batch_test, y_batch_test, _ = data.make_batch(conn, cur, 'test', self.batch_size, 'Building')
						test_accuracy = sess.run(self.accuracy, feed_dict={self.x_image:x_batch_test, self.y_:y_batch_test, self.keep_prob: 1.0})
						print('Test Accuracy:')
						print(test_accuracy)

					f_log.write('%d,%d,%f,%f,%f\n' % (i, k, train_xe, train_accuracy, test_accuracy))
				if k%self.lr_decay_freq == 0:
					self.lr_value = self.lr_value * 0.1
					print('Learning rate:')
					print(self.lr_value)

				summary, _ = sess.run([merged, self.train_step], feed_dict={self.x_image: x_batch, self.y_: y_batch, self.lr:self.lr_value, self.m:self.m_value, self.keep_prob: 0.5})
				#sess.run(self.train_step, feed_dict={self.x_image: x_batch, self.y_: y_batch, self.lr:self.lr_value, self.m:self.m_value, self.keep_prob: 0.5})
				train_writer.add_summary(summary, k)
				k = k + 1

		cur.close()
		conn.close()

		save_path = saver.save(sess, "trained_model/%s/Drone_CNN.ckpt" % self.model_name)
		print('Model saved in file: %s' % save_path)
		#train_writer.close()
		f_log.close()

	def run_prediction(self, sess, x_batch_drone, y_conv_argmax, y_conv_softmax, mode, interest_label, prob_list):
		if mode == 'MOST_PROBABLE_CLASS':
			pred_result = sess.run(y_conv_argmax, feed_dict={self.x_image:x_batch_drone, self.keep_prob: 1.0})
			for prob in pred_result:
				prob_list.append(prob)
		elif mode == 'PROB_OF_INTEREST':
			if interest_label == 'Building':
				interest_ch = 0
			else:
				interest_ch = 1
			pred_result = sess.run(y_conv_softmax, feed_dict={self.x_image:x_batch_drone, self.keep_prob: 1.0})
			for prob in pred_result:
				prob_list.append(prob[interest_ch])

	def drone_prediction(self, mode, window_sliding_stride=8, interest_label='Building'):
		conn, cur = data.get_db_connection()
		drone_dir = data.get_drone_dir_all()

		y_conv_argmax = tf.argmax(self.y_conv, 1)
		y_conv_softmax= tf.nn.softmax(self.y_conv)

		with self.restore() as sess:
			#for each drone ortho-images
			for row in drone_dir:
				print('Current Dataset: %s (%s)' % (row[0], mode))
				curr_image = cv2.imread(row[1])

				x_batch_drone = []

				k = 0
				prob_list = []

				curr_image_h = len(curr_image)
				curr_image_w = len(curr_image[0])

				result_image_h = curr_image_h - curr_image_h%window_sliding_stride - self.input_patch_size
				result_image_w = curr_image_w - curr_image_w%window_sliding_stride - self.input_patch_size

				#For each patch...
				for i in range(0, result_image_h, window_sliding_stride):
					for j in range(0, result_image_w, window_sliding_stride):
						k = k + 1

						patch = np.array(curr_image[i:i+self.input_patch_size, j:j+self.input_patch_size])

						x_batch_drone.append(patch)

						#Run tensorflow with 128 patches
						if k%self.batch_size==0:
							self.run_prediction(sess, x_batch_drone, y_conv_argmax, y_conv_softmax, mode, interest_label, prob_list)
							x_batch_drone = []

				#Run prediction for rest of data
				self.run_prediction(sess, x_batch_drone, y_conv_argmax, y_conv_softmax, mode, interest_label, prob_list)

				result = np.reshape(prob_list, (int(result_image_h/window_sliding_stride), int(result_image_w/window_sliding_stride)))

				if mode == 'PROB_OF_INTEREST':
					result = cv2.resize(result, (len(curr_image[0]), len(curr_image)), interpolation=cv2.INTER_LINEAR) * 255

				cv2.imwrite('result_%s_%s.png' % (row[0], mode), result)
				print('Prediction Complete.')

class Common_image(Common):
	def __init__(self, model_name, input_patch_size, output_patch_size, lr_value, lr_decay_rate, lr_decay_freq, m_value, batch_size):
		Common.__init__(self, model_name, input_patch_size, lr_value, lr_decay_rate, lr_decay_freq, m_value, batch_size)
		self.output_patch_size = output_patch_size
		self.y_ = tf.placeholder('float', shape=[None, output_patch_size, output_patch_size, 3])

	def train(self, epoch):
		os.makedirs('tb/%s' % self.model_name)
		os.makedirs('trained_model/%s' % self.model_name)

		tf.summary.image('input x_image', self.x_image)
		tf.summary.image('y_ground_truth', self.y_)
		tf.summary.image('y_prediction', self.y_conv)
		tf.summary.image('y_pred_softmax', self.y_soft)
		tf.summary.scalar('cross_entropy', self.cross_entropy)
		tf.summary.scalar('learning rate', self.lr)

		sess = tf.Session()

		sess.run(tf.global_variables_initializer())
		sess.run(tf.local_variables_initializer())

		saver = tf.train.Saver()

		f_log = open('trained_model/%s/log.csv' % self.model_name, 'w')

		merged = tf.summary.merge_all()
		train_writer = tf.summary.FileWriter('/home/lsmjn/Drone-CNN/tb/%s' % self.model_name, sess.graph)

		ngii_dir_training = data.get_ngii_dir('training')
		ngii_dir_test = data.get_ngii_dir('test')

		conn, cur = data.get_db_connection()

		steps = data.get_steps(self.batch_size)

		k = 0

		print('\nCurrent Model: %s' % self.model_name)

		for i in range(0, epoch):
			for j in range(0, steps):
				x_batch, _, y_batch = data.make_batch(conn, cur, 'training', self.batch_size)

				if k%10 == 0:
					print('\nstep %d, epoch %d' % (k, i))
					train_xe = sess.run(self.cross_entropy, feed_dict={self.x_image:x_batch, self.y_:y_batch, self.keep_prob: 1.0})
					print('Train XE:')
					print(train_xe)

					f_log.write('%d,%d,%f\n' % (i, k, train_xe))

				if k%self.lr_decay_freq == 0:
					self.lr_value = self.lr_value * 0.1
					print('Learning rate:')
					print(self.lr_value)

				summary, _ = sess.run([merged, self.train_step], feed_dict={self.x_image: x_batch, self.y_: y_batch, self.lr:self.lr_value, self.m:self.m_value, self.keep_prob: 0.5})
				#sess.run(self.train_step, feed_dict={self.x_image: x_batch, self.y_: y_batch, self.lr:self.lr_value, self.m:self.m_value, self.keep_prob: 0.5})
				train_writer.add_summary(summary, k)
				k = k + 1

		cur.close()
		conn.close()

		save_path = saver.save(sess, "trained_model/%s/Drone_CNN.ckpt" % self.model_name)
		print('Model saved in file: %s' % save_path)
		f_log.close()
		train_writer.close()

	def create_test_patches(self):
		y_conv_argmax = tf.argmax(self.y_conv, 1)
		with self.restore() as sess:
			conn, cur = data.get_db_connection()
			x_batch_test, _, y_batch_test = data.make_batch(conn, cur, 'test', 20)
			y_prediction, test_accuracy = sess.run([y_conv_argmax, self.accuracy], feed_dict={self.x_image:x_batch_test, self.y_:y_batch_test, self.keep_prob: 1.0})

			f, axarr = plt.subplots(2, 10)

			f.suptitle('Test Result of Test Dataset (Accuracy: %f)' % test_accuracy)

			for i in range(0, 20):
				if y_prediction[i] == 0:
					y_label = 'Building'
				elif y_prediction[i] == 1:
					y_label = 'Road'
				elif y_prediction[i] == 2:
					y_label = 'Otherwise'
				else:
					y_label = "???"
				axarr[0, i].imshow(x_batch_test[i]) if i < 10 else axarr[1, i-10].imshow(x_batch_test[i])
				axarr[0, i].set_title(y_label) if i < 10 else axarr[1, i-10].set_title(y_label)

			plt.show()

	def drone_prediction(self):
		y_conv_argmax = tf.argmax(self.y_conv, 1)
		conn, cur = data.get_db_connection()
		drone_dir = data.get_drone_dir_all()

		with self.restore() as sess:
			for i in range(0, len(drone_dir)):
				curr_dataset_name = drone_dir[i][0]
				print('Current Dataset: %s' % curr_dataset_name)
				end_idx = data.get_patch_num(curr_dataset_name)

				for start_idx in range(0, end_idx, 20):
					x_batch_drone = data.make_batch_drone(conn, cur, start_idx, 20)
					y_prediction = sess.run(y_conv_argmax, feed_dict={self.x_image:x_batch_drone, self.keep_prob: 1.0})

					'''
					for pred_result in y_prediction:
						if pred_result == 0:
							y_label = 'Building'
						elif pred_result== 1:
							y_label = 'Road'
						elif pred_result== 2:
							y_label = 'Otherwise'
						else:
							y_label = "???"
						print(y_label)
					'''

					f, axarr = plt.subplots(2, 10)

					f.suptitle('Prediction Result of %s Dataset' % curr_dataset_name)

					for i in range(0, 20):
						if y_prediction[i] == 0:
							y_label = 'Building'
						elif y_prediction[i] == 1:
							y_label = 'Road'
						elif y_prediction[i] == 2:
							y_label = 'Otherwise'
						else:
							y_label = "???"
						axarr[0, i].imshow(x_batch_drone[i]) if i < 10 else axarr[1, i-10].imshow(x_batch_drone[i])
						axarr[0, i].set_title(y_label) if i < 10 else axarr[1, i-10].set_title(y_label)

					plt.show()

class VGG16_label(Common_label):
	def __init__(self, model_name, input_patch_size, lr_value, lr_decay_rate, lr_decay_freq, m_value, batch_size):
		Common_label.__init__(self, model_name, input_patch_size, lr_value, lr_decay_rate, lr_decay_freq, m_value, batch_size)

		#Size: 64*64
		self.W_conv1 = self.weight_variable([3, 3, 3, 64], 678678678)
		self.b_conv1 = self.bias_variable([64])
		self.h_conv1 = tf.nn.relu(self.conv2d(self.x_image, self.W_conv1) + self.b_conv1)

		self.W_conv2 = self.weight_variable([3, 3, 64, 64], 12312323)
		self.b_conv2 = self.bias_variable([64])
		self.h_conv2 = tf.nn.relu(self.conv2d(self.h_conv1, self.W_conv2) + self.b_conv2)

		self.W_conv3 = self.weight_variable([3, 3, 64, 64], 12312323)
		self.b_conv3 = self.bias_variable([64])
		self.h_conv3 = tf.nn.relu(self.conv2d(self.h_conv2, self.W_conv3) + self.b_conv3)

		self.W_conv4 = self.weight_variable([3, 3, 64, 64], 12312323)
		self.b_conv4 = self.bias_variable([64])
		self.h_conv4 = tf.nn.relu(self.conv2d(self.h_conv3, self.W_conv4) + self.b_conv4)
		self.h_pool4 = self.max_pool_2x2_stride(self.h_conv4)

		#Size: 32*32
		self.W_conv5 = self.weight_variable([3, 3, 64, 128], 12312323)
		self.b_conv5 = self.bias_variable([128])
		self.h_conv5 = tf.nn.relu(self.conv2d(self.h_pool4, self.W_conv5) + self.b_conv5)

		self.W_conv6 = self.weight_variable([3, 3, 128, 128], 12312323)
		self.b_conv6 = self.bias_variable([128])
		self.h_conv6 = tf.nn.relu(self.conv2d(self.h_conv5, self.W_conv6) + self.b_conv6)

		self.W_conv7 = self.weight_variable([3, 3, 128, 128], 12312323)
		self.b_conv7 = self.bias_variable([128])
		self.h_conv7 = tf.nn.relu(self.conv2d(self.h_conv6, self.W_conv7) + self.b_conv7)
		self.h_pool7 = self.max_pool_2x2_stride(self.h_conv7)

		#Size: 16*16
		self.W_conv8 = self.weight_variable([3, 3, 128, 256], 12312323)
		self.b_conv8 = self.bias_variable([256])
		self.h_conv8 = tf.nn.relu(self.conv2d(self.h_pool7, self.W_conv8) + self.b_conv8)

		self.W_conv9 = self.weight_variable([3, 3, 256, 256], 12312323)
		self.b_conv9 = self.bias_variable([256])
		self.h_conv9 = tf.nn.relu(self.conv2d(self.h_conv8, self.W_conv9) + self.b_conv9)

		self.W_conv10 = self.weight_variable([3, 3, 256, 256], 12312323)
		self.b_conv10 = self.bias_variable([256])
		self.h_conv10 = tf.nn.relu(self.conv2d(self.h_conv9, self.W_conv10) + self.b_conv10)
		self.h_pool10 = self.max_pool_2x2_stride(self.h_conv10)

		#Size: 8*8
		self.W_conv11 = self.weight_variable([3, 3, 256, 512], 12312323)
		self.b_conv11 = self.bias_variable([512])
		self.h_conv11 = tf.nn.relu(self.conv2d(self.h_pool10, self.W_conv11) + self.b_conv11)

		self.W_conv12 = self.weight_variable([3, 3, 512, 512], 12312323)
		self.b_conv12 = self.bias_variable([512])
		self.h_conv12 = tf.nn.relu(self.conv2d(self.h_conv11, self.W_conv12) + self.b_conv12)

		self.W_conv13 = self.weight_variable([3, 3, 512, 512], 12312323)
		self.b_conv13 = self.bias_variable([512])
		self.h_conv13 = tf.nn.relu(self.conv2d(self.h_conv12, self.W_conv13) + self.b_conv13)
		self.h_pool13 = self.max_pool_2x2_stride(self.h_conv13)

		#Size: 4*4
		self.W_fc14 = self.weight_variable([4*4*512,4096], 345345345)
		self.b_fc14 = self.bias_variable([4096])
		self.h_pool13_flat = tf.reshape(self.h_pool13, [-1, 4*4*512])
		self.h_fc14 = tf.nn.relu(tf.matmul(self.h_pool13_flat, self.W_fc14) + self.b_fc14)

		self.W_fc15 = self.weight_variable([4096, 4096], 4546546)
		self.b_fc15 = self.bias_variable([4096])
		self.h_fc15 = tf.nn.relu(tf.matmul(self.h_fc14, self.W_fc15) + self.b_fc15)

		self.W_fc16 = self.weight_variable([4096, 3], 4546546)
		self.b_fc16 = self.bias_variable([3])
		self.h_fc16 = tf.matmul(self.h_fc15, self.W_fc16) + self.b_fc16

		self.y_conv = tf.reshape(self.h_fc16, [-1, 3])

		self.correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1))
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

		self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y_conv))

		self.train_step = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.cross_entropy)

class Saito_label_bn(Common_label):
	def __init__(self, model_name, input_patch_size, lr_value, lr_decay_rate, lr_decay_freq, m_value, batch_size):
		Common_label.__init__(self, model_name, input_patch_size, lr_value, lr_decay_rate, lr_decay_freq, m_value, batch_size)
		self.output_patch_size = int(input_patch_size / 2)

		#C(64, 9*9/2)
		self.W_conv1 = self.weight_variable([16, 16, 3, 64], 678678678)
		self.b_conv1 = self.bias_variable([64])
		self.h_conv1 = tf.nn.relu(self.batch_norm(self.conv2d_stride(self.x_image, self.W_conv1) + self.b_conv1, 64, self.phase_train))

		#P(2/1)
		self.h_pool1 = self.max_pool_2x2(self.h_conv1)

		#C(128, 7*7/1)
		self.W_conv2 = self.weight_variable([4, 4, 64, 112], 12312323)
		self.b_conv2 = self.bias_variable([112])
		self.h_conv2 = tf.nn.relu(self.batch_norm(self.conv2d(self.h_pool1, self.W_conv2) + self.b_conv2, 112, self.phase_train))

		#C(128, 5*5/1)
		self.W_conv3 = self.weight_variable( [3, 3, 112, 80], 234234234)
		self.b_conv3 = self.bias_variable([80])
		self.h_conv3 = tf.nn.relu(self.batch_norm(self.conv2d(self.h_conv2, self.W_conv3) + self.b_conv3, 80, self.phase_train))

		#FC(4096)
		self.W_fc1 = self.weight_variable([self.output_patch_size*self.output_patch_size*80,4096], 345345345)
		self.b_fc1 = self.bias_variable([4096])
		self.h_conv3_flat = tf.reshape(self.h_conv3, [-1, self.output_patch_size*self.output_patch_size*80])
		self.h_fc1 = tf.nn.relu(tf.matmul(self.h_conv3_flat, self.W_fc1) + self.b_fc1)
		self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.keep_prob)

		#FC(768)
		self.W_fc2 = self.weight_variable([4096, 3], 45456)
		self.b_fc2 = self.bias_variable([3])
		self.h_fc2 = tf.matmul(self.h_fc1, self.W_fc2) + self.b_fc2
		self.h_fc2_drop = tf.nn.dropout(self.h_fc2, self.keep_prob)

		self.y_conv = tf.reshape(self.h_fc2_drop, [-1, 3])

		self.correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1))
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

		self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y_conv))

		self.train_step = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=self.m).minimize(self.cross_entropy)

class Saito_image_bn(Common_image):
	def __init__(self, model_name, input_patch_size, lr_value, lr_decay_rate, lr_decay_freq, m_value, batch_size):
		Common_image.__init__(self, model_name, input_patch_size, int(input_patch_size / 2), lr_value, lr_decay_rate, lr_decay_freq, m_value, batch_size)

		#C(64, 9*9/2)
		self.W_conv1 = self.weight_variable([16, 16, 3, 64], 678678678)
		self.b_conv1 = self.bias_variable([64])
		self.h_conv1 = tf.nn.relu(self.batch_norm(self.conv2d_stride(self.x_image, self.W_conv1) + self.b_conv1, 64, self.phase_train))

		#P(2/1)
		self.h_pool1 = self.max_pool_2x2(self.h_conv1)

		#C(128, 7*7/1)
		self.W_conv2 = self.weight_variable([4, 4, 64, 112], 12312323)
		self.b_conv2 = self.bias_variable([112])
		self.h_conv2 = tf.nn.relu(self.batch_norm(self.conv2d(self.h_pool1, self.W_conv2) + self.b_conv2, 112, self.phase_train))

		#C(128, 5*5/1)
		self.W_conv3 = self.weight_variable( [3, 3, 112, 80], 234234234)
		self.b_conv3 = self.bias_variable([80])
		self.h_conv3 = tf.nn.relu(self.batch_norm(self.conv2d(self.h_conv2, self.W_conv3) + self.b_conv3, 80, self.phase_train))

		#FC(4096)
		self.W_fc1 = self.weight_variable([self.output_patch_size*self.output_patch_size*80,4096], 345345345)
		self.b_fc1 = self.bias_variable([4096])
		self.h_conv3_flat = tf.reshape(self.h_conv3, [-1, self.output_patch_size*self.output_patch_size*80])
		self.h_fc1 = tf.nn.relu(tf.matmul(self.h_conv3_flat, self.W_fc1) + self.b_fc1)
		self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.keep_prob)

		#FC(768)
		self.W_fc2 = self.weight_variable([4096, 4800], 45456)
		self.b_fc2 = self.bias_variable([4800])
		self.h_fc2 = tf.matmul(self.h_fc1, self.W_fc2) + self.b_fc2
		self.h_fc2_drop = tf.nn.dropout(self.h_fc2, self.keep_prob)

		self.y_conv = tf.reshape(self.h_fc2_drop, [-1, 40, 40, 3])
		self.y_soft = tf.nn.softmax(self.y_conv)

		self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y_conv))

		self.train_step = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=self.m).minimize(self.cross_entropy)

class Saito_single_label_bn(Common_single_label):
	def __init__(self, model_name, input_patch_size, lr_value, lr_decay_rate, lr_decay_freq, m_value, batch_size):
		Common_single_label.__init__(self, model_name, input_patch_size, lr_value, lr_decay_rate, lr_decay_freq, m_value, batch_size)
		self.output_patch_size = int(input_patch_size / 2)

		#C(64, 9*9/2)
		self.W_conv1 = self.weight_variable([16, 16, 3, 64], 678678678)
		self.b_conv1 = self.bias_variable([64])
		self.h_conv1 = tf.nn.relu(self.batch_norm(self.conv2d_stride(self.x_image, self.W_conv1) + self.b_conv1, 64, self.phase_train))

		#P(2/1)
		self.h_pool1 = self.max_pool_2x2(self.h_conv1)

		#C(128, 7*7/1)
		self.W_conv2 = self.weight_variable([4, 4, 64, 112], 12312323)
		self.b_conv2 = self.bias_variable([112])
		self.h_conv2 = tf.nn.relu(self.batch_norm(self.conv2d(self.h_pool1, self.W_conv2) + self.b_conv2, 112, self.phase_train))

		#C(128, 5*5/1)
		self.W_conv3 = self.weight_variable( [3, 3, 112, 80], 234234234)
		self.b_conv3 = self.bias_variable([80])
		self.h_conv3 = tf.nn.relu(self.batch_norm(self.conv2d(self.h_conv2, self.W_conv3) + self.b_conv3, 80, self.phase_train))

		#FC(4096)
		self.W_fc1 = self.weight_variable([self.output_patch_size*self.output_patch_size*80,4096], 345345345)
		self.b_fc1 = self.bias_variable([4096])
		self.h_conv3_flat = tf.reshape(self.h_conv3, [-1, self.output_patch_size*self.output_patch_size*80])
		self.h_fc1 = tf.nn.relu(tf.matmul(self.h_conv3_flat, self.W_fc1) + self.b_fc1)
		self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.keep_prob)

		#FC(768)
		self.W_fc2 = self.weight_variable([4096, 2], 45456)
		self.b_fc2 = self.bias_variable([2])
		self.h_fc2 = tf.matmul(self.h_fc1, self.W_fc2) + self.b_fc2
		self.h_fc2_drop = tf.nn.dropout(self.h_fc2, self.keep_prob)

		self.y_conv = tf.reshape(self.h_fc2_drop, [-1, 2])


		self.correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1))
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

		self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y_conv))

		self.train_step = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.cross_entropy)
