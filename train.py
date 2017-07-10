import tensorflow as tf
import data
import datetime
import csv

input_patch_size = 64
output_patch_size = 32

x_image = tf.placeholder('float', shape=[None, input_patch_size, input_patch_size, 3])
y_ = tf.placeholder('float', shape=[None, output_patch_size, output_patch_size, 3])

def weight_variable(shape, seed):
	initial = tf.truncated_normal(shape, stddev=0.001, seed=seed)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	#initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def conv2d_stride(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')

#C(64, 9*9/2)
W_conv1 = weight_variable([16, 16, 3, 64], 678678678)
b_conv1 = bias_variable([64])
h_conv1 = tf.nn.relu(conv2d_stride(x_image, W_conv1) + b_conv1)

#P(2/1)
h_pool1 = max_pool_2x2(h_conv1)

#C(128, 7*7/1)
W_conv2 = weight_variable([4, 4, 64, 112], 12312323)
b_conv2 = bias_variable([112])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

#C(128, 5*5/1)
W_conv3 = weight_variable([3, 3, 112, 80], 234234234)
b_conv3 = bias_variable([80])
h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3) + b_conv3)

#FC(4096)
W_fc1 = weight_variable([output_patch_size*output_patch_size*80,4096], 345345345)
b_fc1 = bias_variable([4096])
h_conv3_flat = tf.reshape(h_conv3, [-1, output_patch_size*output_patch_size*80])
h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#FC(768)
W_fc2 = weight_variable([4096, output_patch_size*output_patch_size*3], 45456)
b_fc2 = bias_variable([output_patch_size*output_patch_size*3])
h_fc2 = tf.matmul(h_fc1, W_fc2) + b_fc2
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)


y_conv = tf.reshape(h_fc2_drop, [-1, output_patch_size, output_patch_size, 3])

threshold = tf.constant(0.8, shape=[output_patch_size*output_patch_size*3])
y_conv_threshold = tf.reshape(tf.maximum(tf.zeros(tf.shape(threshold)), tf.add(h_fc2_drop, -threshold)) * 5, [-1, output_patch_size, output_patch_size, 3])

tf.summary.image('input x_image', x_image)
tf.summary.image('input y_', y_)
tf.summary.image('predicted y_conv', y_conv)
tf.summary.image('thresholded y_conv', y_conv_threshold)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
#cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
tf.summary.scalar('cross_entropy', cross_entropy)

lr = tf.placeholder(tf.float32)
m = tf.placeholder(tf.float32)
#train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
#train_step = tf.train.RMSPropOptimizer(lr).minimize(cross_entropy)
train_step = tf.train.MomentumOptimizer(learning_rate=lr, momentum=m).minimize(cross_entropy)

sess = tf.Session()

sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

saver = tf.train.Saver()

f_log = open('trained_model/log.csv', 'w')

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('/home/lsmjn/tensorOrtho_TY/tb', sess.graph)

ngii_dir = data.get_ngii_dir()

num_data = len(data.get_ngii_dir())
lr_value = 0.1 / 0.1
m_value = 0.9
steps = 2

k = 0

for epoch in range(0, 10000000000000000000000000000000000000):
	for i in range(num_data):
		dataset_name = ngii_dir[i][0]
		print('Current Dataset: %s (num_data %d)' % (dataset_name, i))

		for j in range(0, steps):
			x_batch, y_batch,_ = data.make_batch(dataset_name, 8)

			print('step %d, acc step %d, epoch %d' % (j, k, epoch))

			if k%10 == 0:
				train_xe = sess.run(cross_entropy, feed_dict={x_image:x_batch, y_:y_batch, keep_prob:1.0})
				print('Train XE: %f' % train_xe)
				f_log.write('%d,%f\n' % (k, train_xe))
			if k%1000 == 0:
				lr_value = lr_value * 0.1
				print('Learning rate:')
				print(lr_value)

			summary, _ = sess.run([merged, train_step], feed_dict={x_image: x_batch, y_: y_batch, keep_prob:0.5, lr:lr_value, m:m_value})

			train_writer.add_summary(summary, k)

			k = k + 1

save_path = saver.save(sess, "trained_model/ngii_CNN.ckpt")
print('Model saved in file: %s' % save_path)
train_writer.close()
f_log.close()
