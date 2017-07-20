import tensorflow as tf

class Common_label:
    def __init__(self, input_patch_size, lr_value, lr_decay_rate, lr_decay_freq, m_value, batch_size):
        self.input_patch_size = input_patch_size
        self.lr_value = lr_value / lr_decay_rate
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_freq = lr_decay_freq
        self.m_value = m_value
        self.batch_size = batch_size

        self.x_image = tf.placeholder('float', shape=[None, input_patch_size, input_patch_size, 3])
        self.y_ = tf.placeholder('float', shape=[None, 3])
        self.phase_train = tf.Variable(True)
        self.lr = tf.placeholder(tf.float32)
        self.m = tf.placeholder(tf.float32)
        self.keep_prob = tf.placeholder(tf.float32)

    def weight_variable(self, shape, seed):
    	initial = tf.truncated_normal(shape, stddev=0.1, seed=seed)
    	return tf.Variable(initial)

    def bias_variable(self, shape):
    	initial = tf.constant(0.1, shape=shape)
    	#initial = tf.truncated_normal(shape, stddev=0.1)
    	return tf.Variable(initial)

    def conv2d(self, x, W):
    	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def conv2d_stride(self, x, W):
    	return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')

    def max_pool_2x2(self, x):
    	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2_stride(self, x):
    	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def batch_norm(self, x, n_out, phase_train):
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

class VGG16_label(Common_label):
    def __init__(self, input_patch_size, lr_value, lr_decay_rate, lr_decay_freq, m_value, batch_size):
        Common_label.__init__(self, input_patch_size, lr_value, lr_decay_rate, lr_decay_freq, m_value, batch_size)

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

        self.train_step = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=self.m).minimize(self.cross_entropy)



class Saito_label_bn(Common_label):
    def __init__(self, input_patch_size, lr_value, lr_decay_rate, lr_decay_freq, m_value, batch_size):
        Common_label.__init__(self, input_patch_size, lr_value, lr_decay_rate, lr_decay_freq, m_value, batch_size)
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
