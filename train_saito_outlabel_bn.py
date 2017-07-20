import tensorflow as tf
import data
import datetime
import csv

input_patch_size = 64
output_patch_size = 32

x_image = tf.placeholder('float', shape=[None, input_patch_size, input_patch_size, 3])
y_ = tf.placeholder('float', shape=[None, 3])

phase_train = tf.Variable(True)

def weight_variable(shape, seed):
	initial = tf.truncated_normal(shape, stddev=0.1, seed=seed)
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

def batch_norm(x, n_out, phase_train):
    """
    Batch normalization on convolutional maps.
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
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

phase_train = tf.Variable(True)

#C(64, 9*9/2)
W_conv1 = weight_variable([16, 16, 3, 64], 678678678)
b_conv1 = bias_variable([64])
h_conv1 = tf.nn.relu(batch_norm(conv2d_stride(x_image, W_conv1) + b_conv1, 64, phase_train))

#P(2/1)
h_pool1 = max_pool_2x2(h_conv1)

#C(128, 7*7/1)
W_conv2 = weight_variable([4, 4, 64, 112], 12312323)
b_conv2 = bias_variable([112])
h_conv2 = tf.nn.relu(batch_norm(conv2d(h_pool1, W_conv2) + b_conv2, 112, phase_train))

#C(128, 5*5/1)
W_conv3 = weight_variable([3, 3, 112, 80], 234234234)
b_conv3 = bias_variable([80])
h_conv3 = tf.nn.relu(batch_norm(conv2d(h_conv2, W_conv3) + b_conv3, 80, phase_train))

#FC(4096)
W_fc1 = weight_variable([output_patch_size*output_patch_size*80,4096], 345345345)
b_fc1 = bias_variable([4096])
h_conv3_flat = tf.reshape(h_conv3, [-1, output_patch_size*output_patch_size*80])
h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#FC(768)
W_fc2 = weight_variable([4096, 3], 45456)
b_fc2 = bias_variable([3])
h_fc2 = tf.matmul(h_fc1, W_fc2) + b_fc2
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

y_conv = tf.reshape(h_fc2_drop, [-1, 3])

tf.summary.image('input x_image', x_image)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('train_accuracy', accuracy)

#tf.summary.image('input y_', y_)
#tf.summary.image('predicted y_conv', y_conv)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
tf.summary.scalar('cross_entropy', cross_entropy)

lr = tf.placeholder(tf.float32)
tf.summary.scalar('learning rate', lr)
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

ngii_dir_training = data.get_ngii_dir('training')
ngii_dir_test = data.get_ngii_dir('test')

conn, cur = data.get_db_connection()

num_data = len(ngii_dir_training)
lr_value = 0.0001 / 0.1
m_value = 0.9
batch_size = 128
steps = data.get_steps(batch_size)

i = 0

for epoch in range(0, 100):
	for j in range(0, steps):
		x_batch, y_batch = data.make_batch(conn, cur, 'training', batch_size)

		if j%10 == 0:
			print('\nstep %d, epoch %d' % (j, epoch))
			train_xe, train_accuracy = sess.run([cross_entropy, accuracy], feed_dict={x_image:x_batch, y_:y_batch, keep_prob: 1.0})
			print('Train XE:')
			print(train_xe)
			print('Train Accuracy:')
			print(train_accuracy)

			for l in range(0, len(ngii_dir_test)):
				dataset_test_name = ngii_dir_test[l][0]
				x_batch_test, y_batch_test = data.make_batch(conn, cur, 'test', batch_size)
				test_accuracy = sess.run(accuracy, feed_dict={x_image:x_batch_test, y_:y_batch_test, keep_prob: 1.0})
				print('Test Accuracy:')
				print(test_accuracy)

			f_log.write('%d,%f,%f,%f\n' % (j, train_xe, train_accuracy, test_accuracy))
		if j%5000 == 0:
			lr_value = lr_value * 0.1
			print('Learning rate:')
			print(lr_value)

		summary, _ = sess.run([merged, train_step], feed_dict={x_image: x_batch, y_: y_batch, lr:lr_value, m:m_value, keep_prob: 0.5})

		train_writer.add_summary(summary, i)

		i = i + 1

cur.close()
conn.close()

save_path = saver.save(sess, "trained_model/ngii_CNN.ckpt")
print('Model saved in file: %s' % save_path)
train_writer.close()
f_log.close()
