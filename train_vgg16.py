import tensorflow as tf
import data
import datetime
import csv

patch_size = 64
output_patch_size = 8

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

def conv2d_stride(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')

def max_pool_2x2_stride(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#Size: 64*64
W_conv1 = weight_variable([3, 3, 3, 64], 678678678)
b_conv1 = bias_variable([64])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

W_conv2 = weight_variable([3, 3, 64, 64], 12312323)
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2_stride(h_conv2)

#Size: 32*32
W_conv3 = weight_variable([3, 3, 64, 128], 12312323)
b_conv3 = bias_variable([128])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)

W_conv4 = weight_variable([3, 3, 128, 128], 12312323)
b_conv4 = bias_variable([128])
h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)

W_conv5 = weight_variable([3, 3, 128, 128], 12312323)
b_conv5 = bias_variable([128])
h_conv5 = tf.nn.relu(conv2d(h_conv4, W_conv5) + b_conv5)

W_conv6 = weight_variable([3, 3, 128, 256], 12312323)
b_conv6 = bias_variable([256])
h_conv6 = tf.nn.relu(conv2d(h_conv5, W_conv6) + b_conv6)

W_conv7 = weight_variable([3, 3, 256, 256], 12312323)
b_conv7 = bias_variable([256])
h_conv7 = tf.nn.relu(conv2d(h_conv6, W_conv7) + b_conv7)

W_conv8 = weight_variable([3, 3, 256, 256], 12312323)
b_conv8 = bias_variable([256])
h_conv8 = tf.nn.relu(conv2d(h_conv7, W_conv8) + b_conv8)

W_conv9 = weight_variable([3, 3, 256, 256], 12312323)
b_conv9 = bias_variable([256])
h_conv9 = tf.nn.relu(conv2d(h_conv8, W_conv9) + b_conv9)

W_conv10 = weight_variable([3, 3, 256, 256], 12312323)
b_conv10 = bias_variable([256])
h_conv10 = tf.nn.relu(conv2d(h_conv9, W_conv10) + b_conv10)
h_pool10 = max_pool_2x2_stride(h_conv10)

#Size: 16*16
W_conv11 = weight_variable([3, 3, 256, 512], 12312323)
b_conv11 = bias_variable([512])
h_conv11 = tf.nn.relu(conv2d(h_pool10, W_conv11) + b_conv11)

W_conv12 = weight_variable([3, 3, 512, 512], 12312323)
b_conv12 = bias_variable([512])
h_conv12 = tf.nn.relu(conv2d(h_conv11, W_conv12) + b_conv12)

W_conv13 = weight_variable([3, 3, 512, 512], 12312323)
b_conv13 = bias_variable([512])
h_conv13 = tf.nn.relu(conv2d(h_conv12, W_conv13) + b_conv13)
h_pool13 = max_pool_2x2_stride(h_conv13)

#Size: 8*8
W_fc14 = weight_variable([output_patch_size*output_patch_size*512,4096], 345345345)
b_fc14 = bias_variable([4096])
h_pool13_flat = tf.reshape(h_pool13, [-1, output_patch_size*output_patch_size*512])
h_fc14 = tf.nn.relu(tf.matmul(h_pool13_flat, W_fc14) + b_fc14)

W_fc15 = weight_variable([4096, 4096], 4546546)
b_fc15 = bias_variable([4096])
h_fc15 = tf.nn.relu(tf.matmul(h_fc14, W_fc15) + b_fc15)

W_fc16 = weight_variable([4096, 3], 4546546)
b_fc16 = bias_variable([3])
h_fc16 = tf.nn.relu(tf.matmul(h_fc15, W_fc16) + b_fc16)

y_conv = tf.reshape(h_fc16, [-1, 3])

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('train_accuracy', accuracy)

tf.summary.image('input x_image', x_image)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
tf.summary.scalar('cross_entropy', cross_entropy)

lr = tf.placeholder(tf.float32)
tf.summary.scalar('learning rate', lr)
m = tf.placeholder(tf.float32)
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

sess = tf.Session()

sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

saver = tf.train.Saver()

f_log = open('trained_model/log.csv', 'w')

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('/home/lsmjn/tensorOrtho_TY/tb', sess.graph)

ngii_dir = data.get_ngii_dir()

num_data = len(data.get_ngii_dir())
lr_value = 0.001 / 0.01
m_value = 0.9
steps = 1965

k = 0

for epoch in range(0, 100):
	for i in range(num_data):
		dataset_name = ngii_dir[i][0]
		print('Current Dataset: %s (num_data %d)' % (dataset_name, i))

		for j in range(0, steps):
			x_batch, y_batch,_,_ = data.make_batch(dataset_name, 128, 'ohe')

			print('step %d, acc step %d, epoch %d' % (j, k, epoch))


			if k%10 == 0:
				train_xe, train_accuracy = sess.run([cross_entropy, accuracy], feed_dict={x_image:x_batch, y_:y_batch})
				print('Train XE: %f, Train Accuracy: %f' % (train_xe, train_accuracy))
				f_log.write('%d,%f,%f\n' % (k, train_xe, train_accuracy))
			if k%1000 == 0:
				lr_value = lr_value * 0.01
				print('Learning rate:')
				print(lr_value)

			summary, _ = sess.run([merged, train_step], feed_dict={x_image: x_batch, y_: y_batch, lr:lr_value, m:m_value})

			train_writer.add_summary(summary, k)

			k = k + 1

save_path = saver.save(sess, "trained_model/ngii_CNN.ckpt")
print('Model saved in file: %s' % save_path)
train_writer.close()
f_log.close()
