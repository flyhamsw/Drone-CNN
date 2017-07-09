import tensorflow as tf
import data
import datetime
import csv

patch_size = 64

x_image = tf.placeholder('float', shape=[None, patch_size, patch_size, 3])
y_ = tf.placeholder('float', shape=[None, 3])

def weight_variable(shape, seed):
	initial = tf.truncated_normal(shape, stddev=1, seed=seed)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	#initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def conv2d_stride2(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')

W_conv1 = weight_variable([3, 3, 3, 64], 678678678)
b_conv1 = bias_variable([64])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([3, 3, 64, 64], 12312323)
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

W_conv3 = weight_variable([3, 3, 64, 64], 12312323)
b_conv3 = bias_variable([64])
h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3) + b_conv3)

W_conv4 = weight_variable([3, 3, 64, 128], 234234234)
b_conv4 = bias_variable([128])
h_conv4 = tf.nn.relu(conv2d_stride2(h_conv3, W_conv4) + b_conv4)
h_pool4 = max_pool_2x2(h_conv4)

W_conv5 = weight_variable([3, 3, 128, 128], 234234234)
b_conv5 = bias_variable([128])
h_conv5 = tf.nn.relu(conv2d(h_pool4, W_conv5) + b_conv5)

W_conv6 = weight_variable([3, 3, 128, 256], 234234234)
b_conv6 = bias_variable([256])
h_conv6 = tf.nn.relu(conv2d_stride2(h_conv5, W_conv6) + b_conv6)
h_pool6 = max_pool_2x2(h_conv6)

W_conv7 = weight_variable([3, 3, 256, 256], 234234234)
b_conv7 = bias_variable([256])
h_conv7 = tf.nn.relu(conv2d(h_pool6, W_conv7) + b_conv7)

#FC(4096)
W_fc1 = weight_variable([16*16*256,4096], 345345345)
b_fc1 = bias_variable([4096])
h_conv7_flat = tf.reshape(h_conv7, [-1, 16*16*256])
h_fc1 = tf.nn.relu(tf.matmul(h_conv7_flat, W_fc1) + b_fc1)
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([4096, 4096], 46466)
b_fc2 = bias_variable([4096])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

#FC(3)
W_fc3 = weight_variable([4096, 3], 45456)
b_fc3 = bias_variable([3])
h_fc3 = tf.matmul(h_fc2_drop, W_fc3) + b_fc3
h_fc3_drop = tf.nn.dropout(h_fc3, keep_prob)

y_conv = tf.reshape(h_fc3_drop, [-1, 3])

#tf.metrics.precision(labels=y_, predictions=y_conv)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('train_accuracy', accuracy)

tf.summary.image('input x_image', x_image)
#tf.summary.image('input y_', y_)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
tf.summary.scalar('cross_entropy', cross_entropy)

lr = tf.placeholder(tf.float32)
tf.summary.scalar('learning rate', lr)
m = tf.placeholder(tf.float32)
train_step = tf.train.RMSPropOptimizer(lr).minimize(cross_entropy)

sess = tf.Session()

sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

saver = tf.train.Saver()

f_log = open('trained_model/log.csv', 'w')

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('/home/lsmjn/tensorOrtho_TY/tb', sess.graph)

ngii_dir = data.get_ngii_dir()

num_data = len(data.get_ngii_dir())
lr_value = 0.1 / 0.01
m_value = 0.9
steps = 247

k = 0

for epoch in range(0, 100):
	for i in range(num_data):
		dataset_name = ngii_dir[i][0]
		print('Current Dataset: %s (num_data %d)' % (dataset_name, i))

		for j in range(0, steps):
			x_batch, y_batch,_,_ = data.make_batch(dataset_name, 256, 'ohe')

			print('step %d, acc step %d, epoch %d' % (j, k, epoch))

			if k%10 == 0:
				train_xe, train_accuracy = sess.run([cross_entropy, accuracy], feed_dict={x_image:x_batch, y_:y_batch, keep_prob:1.0})
				print('Train XE: %f, Train Accuracy: %f' % (train_xe, train_accuracy))
				f_log.write('%d,%f,%f\n' % (k, train_xe, train_accuracy))
			if k%1000 == 0:
				lr_value = lr_value * 0.01
				print('Learning rate:')
				print(lr_value)

			summary, _ = sess.run([merged, train_step], feed_dict={x_image: x_batch, y_: y_batch, keep_prob:0.5, lr:lr_value, m:m_value})

			train_writer.add_summary(summary, k)

			k = k + 1

save_path = saver.save(sess, "trained_model/ngii_CNN.ckpt")
print('Model saved in file: %s' % save_path)
train_writer.close()
f_log.close()
