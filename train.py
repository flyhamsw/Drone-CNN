import tensorflow as tf
import os

import models
import data

def train_model(model, model_name, epoch):
	with tf.variable_scope('model_name'):
		#os.makedirs('tb/%s' % model_name)
		os.makedirs('trained_model/%s' % model_name)

		#tf.summary.image('input x_image', model.x_image)
		#tf.summary.scalar('train_accuracy', model.accuracy)
		#tf.summary.scalar('cross_entropy', model.cross_entropy)
		#tf.summary.scalar('learning rate', model.lr)

		sess = tf.Session()

		sess.run(tf.global_variables_initializer())
		sess.run(tf.local_variables_initializer())

		saver = tf.train.Saver()

		f_log = open('trained_model/%s/log.csv' % model_name, 'w')

		#merged = tf.summary.merge_all()
		#train_writer = tf.summary.FileWriter('/home/lsmjn/tensorOrtho_TY/tb/%s' % model_name, sess.graph)

		ngii_dir_training = data.get_ngii_dir('training')
		ngii_dir_test = data.get_ngii_dir('test')

		conn, cur = data.get_db_connection()

		steps = data.get_steps(batch_size)

		k = 0

		print('\nCurrent Model: %s' % model_name)

		for i in range(0, epoch):
			for j in range(0, steps):
				x_batch, y_batch = data.make_batch(conn, cur, 'training', batch_size)

				if j%10 == 0:
					print('\nstep %d, epoch %d' % (j, i))
					train_xe, train_accuracy = sess.run([model.cross_entropy, model.accuracy], feed_dict={model.x_image:x_batch, model.y_:y_batch, model.keep_prob: 1.0})
					print('Train XE:')
					print(train_xe)
					print('Train Accuracy:')
					print(train_accuracy)

					for l in range(0, len(ngii_dir_test)):
						dataset_test_name = ngii_dir_test[l][0]
						x_batch_test, y_batch_test = data.make_batch(conn, cur, 'test', batch_size)
						test_accuracy = sess.run(model.accuracy, feed_dict={model.x_image:x_batch_test, model.y_:y_batch_test, model.keep_prob: 1.0})
						print('Test Accuracy:')
						print(test_accuracy)

					f_log.write('%d,%f,%f,%f\n' % (j, train_xe, train_accuracy, test_accuracy))
				if j%5000 == 0:
					model.lr_value = model.lr_value * 0.1
					print('Learning rate:')
					print(model.lr_value)

				#summary, _ = sess.run([merged, model.train_step], feed_dict={model.x_image: x_batch, model.y_: y_batch, model.lr:model.lr_value, model.m:model.m_value, model.keep_prob: 0.5})
				sess.run(model.train_step, feed_dict={model.x_image: x_batch, model.y_: y_batch, model.lr:model.lr_value, model.m:model.m_value, model.keep_prob: 0.5})
				#train_writer.add_summary(summary, k)
				k = k + 1

		cur.close()
		conn.close()

		save_path = saver.save(sess, "trained_model/%s/Drone_CNN.ckpt" % model_name)
		print('Model saved in file: %s' % save_path)
		#train_writer.close()
		f_log.close()

if __name__=='__main__':
	batch_size = 128
	model_dict = {'Saito_label_bn': models.Saito_label_bn(input_patch_size=64, lr_value=0.0001, lr_decay_rate=0.1, lr_decay_freq=5000, m_value=0.9, batch_size=batch_size), 'VGG16_label': models.VGG16_label(input_patch_size=64, lr_value=0.0005, lr_decay_rate=0.1, lr_decay_freq=5000, m_value=0.9, batch_size=batch_size)}
	#model_dict = {'VGG16_label': models.VGG16_label(input_patch_size=64, lr_value=0.0005, lr_decay_rate=0.1, lr_decay_freq=1000, m_value=0.9, batch_size=batch_size), 'Saito_label_bn': models.Saito_label_bn(input_patch_size=64, lr_value=0.0001, lr_decay_rate=0.1, lr_decay_freq=5000, m_value=0.9, batch_size=batch_size)}

	for model_name in model_dict:
		train_model(model_dict[model_name], model_name, epoch=50)
